# Local image dataset

This folder contains the current 3,000-image research sample. It does not contain CIFAKE.

| Split | photo | ai_generated | Total |
|---|---:|---:|---:|
| train | 1050 | 1050 | 2100 |
| val | 225 | 225 | 450 |
| test | 225 | 225 | 450 |

`manifest.csv` stores relative paths, labels, fixed split assignments, group IDs,
pixel hashes, original dimensions, source and generator identifiers.
Keep it together with `images/`. Do not regenerate the split for each training run.
Original images have at least 225 pixels on each side; the loader resizes them to 224x224.

## Sources and attribution

- Tiny-GenImage: https://huggingface.co/datasets/TheKernel01/Tiny-GenImage
  The dataset card declares CC-BY-NC-SA-4.0. Real photos and generated images come
  from a subset of its training shards. This is not its official benchmark split.
- Original GenImage research: https://github.com/GenImage-Dataset/GenImage
  Zhu et al., GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image (2023).

## Sampling and limitations

Seed 42, 1,500 images per class. Exact decoded-pixel duplicates were removed.
Existing photo/AI split assignments were preserved when the cartoon class was removed.
Near-duplicates, related compositions and generator leakage are not fully addressed.
This dataset evaluates photographs versus AI-generated images, not arbitrary artwork.
More data does not establish reliable detection on unseen generators.

The downloader and source archives were removed after preparing this fixed sample.
To train on another computer, transfer images/ and manifest.csv separately from Git.
Preserve this file for source attribution and reproducibility.

## Source revisions and archive hashes

Tiny-GenImage revision: `89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69`

- tiny_train_00000.parquet: SHA-256 `27cfe6c9f36c97246107bd7e54f0ddf55c8584833d7843d96f6674e6536140ce`
- tiny_train_00001.parquet: SHA-256 `82b6605e8d46610a6c02e1e294610efea335fe5778ce97e3bdf0d727446aa262`
