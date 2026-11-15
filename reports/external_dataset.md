# External evaluation dataset preparation

Date: 2026-11-03.
Source: https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval
Pinned revision: 7d4a74a88d2cac93b513c0853bf92c260eaceea0.
Source labels: 0 = real; 1 = generated. Original repository: 51,836 rows,
approximately 206 GB, stored in 413 Parquet shards.

## Fixed protocol

Preparation completed locally on 2026-11-03. Actual contents: 817 COCO photos,
183 FFHQ photos, and 200 generated images from each of the five listed generators.
All 2,000 entries are marked test. The screening excluded four exact duplicates,
nine perceptual-duplicate candidates, and one image below the size threshold.
These exclusion counts combine comparisons against the original data and accepted
external images; they do not imply all exclusions were overlaps with training.
Stored image files total 624,965,636 bytes (approximately 625 MB).

External manifest SHA-256:
`86541f2c662ffe4876ef896ad9e6637704f6041aa4328aa6850448823b4851b7`.
Original manifest used for overlap screening:
`ded5b4dd615ec18d6f2f862b586a4a11a016bca60773e3bdea0ba19b3eba6da2`.

The transfer archive is transfer/community_forensics_external_2000.zip. Its paths
start with data/external_community; extract it from the project root on LiU.
The original dataset remains unchanged. No selected-model evaluation has run yet.

Target 2,000 images: 1,000 real-source photos (COCO/FFHQ) and 1,000 generated
images split equally across DFGAN, FLUX-dev, FLUX-schnell, Firefly_Image2 and
Firefly_Image3. Seed: 20261103. This selection is made without model predictions.
The file preparation script records actual counts, exclusions, source distribution
and hashes in data/external_community/provenance.json on successful completion.

To bound transfer cost, sample homogeneous Parquet row groups with compressed
image columns no larger than 32 MiB. Randomize eligible groups with the fixed seed,
visit source-pair buckets round-robin and shuffle images within each group.
Exclude flagged NSFW entries, undecodable images and images with shortest side
below 224 pixels. Keep original encoded bytes without class-dependent conversion.
Pixel hashes use EXIF-oriented RGB decoding. Exclude exact duplicates and dHash
matches at Hamming distance <=4 against all original 3,000 images and all previously
accepted external images. This perceptual filter is heuristic, not a complete
near-duplicate or shared-subject audit; it can reject different but similar images.

## Evaluation plan

Use the fixed original-validation-selected trial 4 checkpoint from study
20261102_084414_406073. Do not tune weights, thresholds or preprocessing on this
external set. Report overall accuracy, macro F1 and class precision/recall, AI
recall separately for each generator, and photo recall by source. Preserve the
checkpoint hash, manifest hash and every prediction. No evaluation of that remote
checkpoint has yet been performed as part of dataset preparation.

## Scope and caveats

This is a custom source-shift evaluation, not a representative random sample of
the full Community Forensics benchmark. Row-group selection, size constraints and
five-generator restriction introduce sampling bias. The real-source composition
is not matched to every synthetic source pair. Encodings/resolutions can remain
correlated with class; source artifacts can affect detection. A lower score cannot
be attributed solely to generator novelty. Labels are inherited from the provider,
not independent human verification of every example. FFHQ can include unusual
face depictions; manual scope inspection remains advisable for a strict photo task.

Different source names and our duplicate screen do not prove absence of overlap
with ImageNet pretraining or related images in Tiny-GenImage. This set does not
validate mixed posters, small localized AI edits or general deployment readiness.

Once its results influence development, treat it as a development benchmark and
reserve a further untouched set for later final claims. Do not train on this set.

## Attribution

The dataset card specifies non-commercial research and educational use,
CC-BY-NC-SA-4.0, and applicable generator/source terms. Preserve SOURCE_README.md.
Cite Park and Owens, Community Forensics: Using Thousands of Generators to Train
Fake Image Detectors, CVPR 2025; Lin et al., Microsoft COCO, ECCV 2014;
Karras et al., A Style-Based Generator Architecture for GANs, CVPR 2019 (FFHQ).
No images are committed to Git; preparation code and this protocol are versioned.
