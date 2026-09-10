# Anatomical vertebrae refinement

This optional engine integrates Yixiao Liu's standalone warm-up program without
changing its algorithm or thresholds. It applies sequence correction, fragment
cleanup, and adjacent-level boundary refinement to prediction masks. It needs
no CT image, model weights, training, or GPU.

## Configuration and input

Keep the existing `class_map` and set these fields in `config.yaml`:

```yaml
target_organs:
  - vertebrae
vertebrae_engine: anatomical_refinement
affine_reference_file_name: vertebrae_L5.nii.gz
if_save_combined_label: true
```

The reference must be an existing mask on the same grid as the other masks.
For partial scans without an L5 file, choose another existing vertebral mask.
Input cases contain `segmentations/vertebrae_L5.nii.gz`, and similarly named
files through `vertebrae_C1.nii.gz`. Masks must be mutually exclusive, finite,
three-dimensional, and on the same physical grid. Align inconsistent grids
before using ShapeKit; the adapter does not resample them. The existing loader
is not a substitute for checking equal affines across input files.

The adapter combines masks by name into internal labels 1=L5 through 24=C1,
passes the actual reference affine to the unchanged core, and returns binary
masks to ShapeKit. ShapeKit's combined output uses its configured class map,
normally 26=L5 through 49=C1. Do not compare these integer maps without mapping
their label IDs. Other organ entries are left unchanged by this adapter.

## Run

From the repository root, in an environment with ShapeKit's dependencies:

```bash
python main.py --input_folder /path/to/input --output_folder /path/to/new_output --cpu_count 1 --log_folder /path/to/logs
```

Use separate input and output directories and a fresh output directory for the
initial run. Start with one worker. Each worker holds several 3D arrays and
per-vertebra masks, so increase `--cpu_count` only after measuring available
memory and single-case resource use. This engine introduces no dependencies
beyond NumPy, SciPy, and NiBabel, which ShapeKit already requires.

Inspect both output files and the logs; ShapeKit catches case exceptions, so
a successful process exit alone does not establish that every case succeeded.
Avoid relying on `--continue_prediction` after an interrupted run: the existing
resume check only tests for a nonempty segmentation directory, not completeness.
The shared writer now writes supplied empty masks too, preventing copied input
foreground from surviving when refinement removes an entire small component.

## Tests

```bash
python -m unittest discover -s tests -p 'test_vertebrae_refinement.py' -v
```

The tests cover label conversion, affine forwarding, unchanged non-vertebral
entries, empty input, a partial-scan core comparison, and rejection of invalid
or overlapping inputs. These are software tests, not segmentation accuracy
measurements. Missing vertebral masks are allowed; the unchanged core decides
whether the available anatomy supports an identity correction. Partial-scan
support does not establish clinical accuracy on partial scans.

The source program is preserved in `utils/vertebrae_refinement_core.py`.
The small ShapeKit adapter is `utils/vertebrae_refinement.py`. No experiment
images, prediction volumes, or private data are included in this contribution.

## Local integration validation

Seven unit tests passed. The CLI was also tested with one and two workers on
Case06 and Case31 prediction masks. Inputs retained all foreground and native
voxel spacing in a bounding-box crop with a corrected affine. Case06 used a raw
prediction; Case31 used an archived first-submission prediction. This checks
integration equivalence, not segmentation accuracy or a new DSC estimate.

| Check | Case06 | Case31 |
| --- | ---: | ---: |
| Tested array shape | 210 x 254 x 276 | 113 x 201 x 863 |
| ShapeKit vs standalone differing voxels after label mapping | 0 | 0 |
| Two workers vs one worker differing voxels | 0 | 0 |
| Individual masks verified against standalone output | 24 | 24 |

Output affines matched the reference images. The core file is byte-identical
to the supplied standalone program, with SHA-256
`620ce03a05b6d19768e9131e761142e312720da8e6a8c91f6044069d2251f782`.
Testing used Python 3.9, NumPy 2.0.2, SciPy 1.13.1, and NiBabel 5.3.3 on CPU.
Full-field-of-view memory requirements and large-cohort throughput were not
benchmarked in this integration check.
