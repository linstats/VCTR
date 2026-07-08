# GRAPE publication figures

This directory contains reproducible code for descriptive manuscript figures. It reads the existing audit tables and Level-2 image tensors; it does not create a separate image preprocessing path.

## Representative paired visit

Rank all primary-analysis pairs by robust distance from the median age and mean IOP, with a smaller penalty for the paired-eye IOP difference:

```bash
python src/experiments/grape/figures/select_representative_pair.py
```

The complete ranked audit table is written to:

```text
src/experiments/grape/runs/figures/paired_image_partitions_v1/candidate_pairs.csv
```

The configured manuscript example is pair `24_1.53214774282` (subject 24, visit 4). It ranks first under the stated numerical rule and passed visual QC for both CFP and ROI images.

## Paired CFP/ROI partition figure

Generate the four panels and the composite PDF/PNG:

```bash
python src/experiments/grape/figures/plot_image_partitions.py \
  --config src/experiments/grape/configs/figures/paired_image_partitions_v1.json
```

Outputs are written under:

```text
src/experiments/grape/outputs/figures/paired_image_partitions_v1/
```

The current partitions are the full-CV-selected configurations: CFP uses `3x4x1`, and ROI uses `6x2x1`. The OS panels are horizontally flipped because the figure reads the same Level-2 tensors used by the model.

## Publication-style coefficient functions

For an aggregated coefficient-bootstrap run, generate a central-age main
figure and a full-range supplement figure with:

```bash
python src/experiments/grape/figures/plot_at_publication.py \
  --config src/experiments/grape/configs/coefficient_bootstrap/roi_x_vf_pca_gender_patient_pilot_b500.json
```

Both PNG/PDF pairs and their plotting metadata are written into the same
bootstrap run's `figures/` directory. The script uses documented main age/y
defaults unless the config provides an optional `publication_figure` section;
no model or bootstrap refit is performed.

For a coefficient-only panel without bootstrap bands, use
`plot_at_line_only.py`. The layout is configurable and defaults to 4 rows by
3 columns; blocks are placed in numerical row-major order.
