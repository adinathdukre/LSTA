
# Adapt Where Patients Differ: Learned Subspace Test-Time Adaptation for Medical Time Series


## Introduction

LSTA adapts a pretrained medical time-series classifier to a new subject using a few unlabeled windows. It learns a shared low-rank space of LayerNorm updates from training subjects while keeping the backbone frozen. At test time, adaptation is restricted to this learned space.

## Usage

### 1. Install requirements

```bash
pip install -r requirements.txt
```

The code requires PyTorch 2.x. For GPU execution, install a PyTorch build compatible with your CUDA environment.

### 2. Prepare data

Download the datasets from [Medformer](https://github.com/DL4mHealth/Medformer) and follow TeCh's preprocessing and subject-independent split protocol. TDBRAIN requires permission from the dataset provider.

Place the prepared datasets in their corresponding folders:

```text
./data/ADFTD/
./data/APAVA/
./data/TDBRAIN/
./data/PTB/
./data/PTB-XL/
```

Each dataset folder should contain `Feature/` and `Label/label.npy`, following the format expected by the data loaders.

### 3. Download checkpoints

Download the pretrained TeCh backbone checkpoints from [Hugging Face](https://huggingface.co/springlake49/LSTA/tree/main).

Place the downloaded `checkpoints/` folder in the repository root, preserving its directory structure:

```text
./checkpoints/baseline/<DATASET>/<experiment_name>/checkpoint.pth
```

These are source backbone checkpoints. The experiment scripts train and save the learned LSTA adaptation bases separately.

### 4. Run experiments

Dataset-specific scripts are provided under `./scripts/`. Run commands from the repository root.

For example, run APAVA with:

```bash
bash ./scripts/APAVA.sh
```

To select a GPU:

```bash
GPU=1 bash ./scripts/APAVA.sh
```

Run the other datasets with:

```bash
bash ./scripts/ADFTD.sh
bash ./scripts/TDBRAIN.sh
bash ./scripts/PTB.sh
bash ./scripts/PTB-XL.sh --tune_subjects 400
```

The scripts train the adaptation basis, select its configuration using validation macro-F1, and evaluate on test subjects. Matching saved basis checkpoints are reused automatically.

### 5. View results

Training records, learned bases, and evaluation results are saved under:

```text
./runs/logs/TTA_SWEEP/<DATASET>/
```

Per-seed results are saved in `test_result.json` and `test_result.txt`. Aggregated results are saved in `summary.json` and `summary.txt`.

## Acknowledgements

This project builds on TeCh and [Medformer](https://github.com/DL4mHealth/Medformer). We thank their authors for sharing their code and preprocessing resources.
