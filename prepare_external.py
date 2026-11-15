"""Create a bounded, reproducible external research evaluation subset.

No training or prediction is performed. Original dataset is pinned by revision.
"""
import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests
from PIL import Image, ImageOps
from tqdm import tqdm
from external_data_tools import RangeFile

ROOT = Path(__file__).resolve().parent
REPO = 'OwensLab/CommunityForensics-Eval'
REVISION = '7d4a74a88d2cac93b513c0853bf92c260eaceea0'
BASE = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}'
DESTINATION = ROOT / 'data' / 'external_community'
GENERATORS = ('DFGAN', 'FLUX-dev', 'FLUX-schnell', 'Firefly_Image2', 'Firefly_Image3')


def pixel_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def difference_hash(image):
    pixels = list(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS).getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y*9+x] > pixels[y*9+x+1])
    return value


def scan_shard(index):
    import pyarrow.parquet as pq
    name = f'data/CompEval-{index:05d}-of-00413.parquet'
    with RangeFile(f'{BASE}/{name}') as remote:
        parquet = pq.ParquetFile(remote)
        groups = []
        for group in range(parquet.num_row_groups):
            meta = parquet.metadata.row_group(group)
            columns = {meta.column(i).path_in_schema: meta.column(i) for i in range(meta.num_columns)}
            def single(key):
                stats = columns[key].statistics
                return stats.min if stats and stats.has_min_max and stats.min == stats.max else None
            groups.append({'shard': index, 'group': group, 'rows': meta.num_rows,
                           'model': single('model_name'), 'label': single('label'),
                           'source': single('real_source'),
                           'image_bytes': columns['image_data'].total_compressed_size})
    return groups


def old_hashes(manifest):
    # Recompute from actual decoded originals to use the same hash convention.
    hashes, perceptual = set(), []
    with manifest.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    for row in tqdm(rows, desc='Checking existing images'):
        with Image.open(manifest.parent / row['path']) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB')
            hashes.add(pixel_hash(rgb))
            perceptual.append(difference_hash(rgb))
    return hashes, perceptual


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-class', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--index-only', action='store_true')
    parser.add_argument('--max-download-gb', type=float, default=8)
    args = parser.parse_args()
    if args.per_class < 1 or args.per_class % len(GENERATORS):
        parser.error('--per-class must be positive and divisible by five')
    DESTINATION.mkdir(parents=True, exist_ok=True)
    metadata_file = DESTINATION / 'source_index.json'
    if not metadata_file.exists():
        with ThreadPoolExecutor(max_workers=8) as pool:
            shards = list(tqdm(pool.map(scan_shard, range(413)), total=413, desc='Reading metadata'))
        metadata_file.write_text(json.dumps({'revision': REVISION, 'groups': sum(shards, [])}), encoding='utf-8')
    index = json.loads(metadata_file.read_text(encoding='utf-8'))
    if index['revision'] != REVISION:
        raise ValueError('Metadata revision mismatch')
    groups = index['groups']
    print('Indexed models:', sorted({g['model'] for g in groups if g['model']}))
    print('Sources:', sorted({g['source'] for g in groups if g['source']}))
    if args.index_only:
        return
    manifest = DESTINATION / 'manifest.csv'
    if manifest.exists():
        raise FileExistsError('External manifest already exists; refusing to replace the fixed evaluation set')
    # Eligibility is defined by metadata before any model prediction. Restrict photos
    # to COCO/FFHQ and select five AI generators with manageable row groups. This is a
    # custom subset, not the full benchmark. Preserve this filter in provenance.
    candidates = [g for g in groups if g['label'] in (0, 1) and g['model'] and
                  ((g['label'] == 0 and str(g['source']).lower() in ('coco', 'ffhq')) or
                   (g['label'] == 1 and g['model'] in GENERATORS)) and
                  g['image_bytes'] <= 32 * 1024**2]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    # Round-robin generator/source groups to avoid taking only the first generator.
    buckets = {}
    for group in candidates:
        buckets.setdefault((group['label'], group['model']), []).append(group)
    ordered = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]:
                ordered.append(buckets[key].pop())
    used, perceptual = old_hashes(ROOT / 'data' / 'manifest.csv')
    rows, counts, downloaded = [], Counter(), 0
    generator_counts = Counter()
    rejected = Counter()
    (DESTINATION / 'images').mkdir(exist_ok=True)
    partial = DESTINATION / 'partial_manifest.csv'
    progress = tqdm(ordered, desc='Downloading selected groups')
    for group in progress:
        label = 'photo' if group['label'] == 0 else 'ai_generated'
        if counts[label] >= args.per_class:
            continue
        if label == 'ai_generated' and generator_counts[group['model']] >= args.per_class // len(GENERATORS):
            continue
        if downloaded + group['image_bytes'] > args.max_download_gb * 1e9:
            raise RuntimeError('Download budget reached. Partial manifest retained; no complete manifest created.')
        name = f"data/CompEval-{group['shard']:05d}-of-00413.parquet"
        with RangeFile(f'{BASE}/{name}') as remote:
            data = pq.ParquetFile(remote).read_row_group(group['group'], columns=[
                'image_data', 'image_name', 'label', 'model_name', 'nsfw_flag', 'real_source']).to_pylist()
        downloaded += group['image_bytes']
        rng.shuffle(data)
        for record in data:
            if counts[label] >= args.per_class:
                break
            if label == 'ai_generated' and generator_counts[group['model']] >= args.per_class // len(GENERATORS):
                break
            if record['label'] != group['label']:
                raise ValueError('Source label does not match metadata')
            if str(record['nsfw_flag']).lower() == 'true':
                rejected['flagged'] += 1
                continue
            payload = record['image_data']
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    fmt = image.format
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
            except (OSError, ValueError):
                rejected['decode'] += 1
                continue
            digest = pixel_hash(rgb)
            if digest in used:
                rejected['exact_duplicate'] += 1
                continue
            if min(rgb.size) < 224:
                rejected['small'] += 1
                continue
            visual_hash = difference_hash(rgb)
            if any((visual_hash ^ previous).bit_count() <= 4 for previous in perceptual):
                rejected['perceptual_duplicate_candidate'] += 1
                continue
            used.add(digest)
            perceptual.append(visual_hash)
            # Store unchanged encoded bytes, avoiding class-dependent re-encoding.
            extension = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp'}.get(fmt, '.img')
            path = f'images/{digest}{extension}'
            (DESTINATION / path).write_bytes(payload)
            rows.append({'path': path, 'label': label, 'split': 'test', 'group_id': digest,
                         'pixel_sha256': digest, 'width': rgb.width, 'height': rgb.height,
                         'source': REPO, 'source_revision': REVISION,
                         'generator': record['model_name'] if label == 'ai_generated' else '',
                         'real_source': record['real_source'], 'source_pair': record['model_name'],
                         'source_image': record['image_name'], 'source_shard': name,
                         'source_row_group': group['group'],
                         'file_sha256': hashlib.sha256(payload).hexdigest()})
            counts[label] += 1
            if label == 'ai_generated':
                generator_counts[group['model']] += 1
        if rows:
            with partial.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        progress.set_postfix(photos=counts['photo'], ai=counts['ai_generated'], target=args.per_class)
        if all(counts[label] == args.per_class for label in ('photo', 'ai_generated')):
            break
    if any(counts[label] != args.per_class for label in ('photo', 'ai_generated')):
        raise RuntimeError(f'Insufficient eligible images: {dict(counts)}; partial results retained')
    partial.replace(manifest)
    provenance = {'repository': REPO, 'revision': REVISION, 'seed': args.seed,
                  'counts': dict(counts), 'rejected': dict(rejected),
                  'compressed_image_column_bytes_requested': downloaded,
                  'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  'existing_manifest_sha256': hashlib.sha256((ROOT/'data'/'manifest.csv').read_bytes()).hexdigest(),
                  'generator_counts': dict(Counter(r['generator'] for r in rows if r['label']=='ai_generated')),
                  'photo_sources': dict(Counter(r['real_source'] for r in rows if r['label']=='photo')),
                  'near_duplicate_audit': 'Conservative 64-bit dHash screening (Hamming distance <=4 excluded) against existing and accepted images. Does not establish absence of cropped or semantic duplicates; manual audit still recommended.',
                  'sampling': 'Seeded row-group sampling, round-robin source-pair buckets; photos from COCO/FFHQ, equal AI quotas for five named generators; homogeneous groups <=32 MiB, min side 224, exclude flagged NSFW',
                  'scope': 'Custom research subset; not a representative full benchmark or guaranteed unseen-generator test'}
    (DESTINATION / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    response = requests.get(f'{BASE}/README.md', timeout=120)
    response.raise_for_status()
    (DESTINATION / 'SOURCE_README.md').write_text(response.text, encoding='utf-8')
    print(f'Completed external dataset: {manifest}; counts: {dict(counts)}')


if __name__ == '__main__':
    main()
