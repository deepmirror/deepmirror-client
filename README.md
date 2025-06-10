# deepmirror

`deepmirror` is a command-line interface for interacting with the [deepmirror API](https://api.app.deepmirror.ai/public/docs). It allows you to train models, run predictions, and submit structure prediction jobs directly from your terminal.

## Installation

```bash
pip install deepmirror
```

## Authentication

Before using most commands, you need to log in to get your API token:

```bash
dm login EMAIL
```

This saves your token and host in `~/.config/deepmirror/` for reuse.

## Model Commands

### List Available Models

```bash
dm model list
```

### View Model Metadata

```bash
dm model metadata MODEL_ID
```

### Get Full Model Info

```bash
dm model info MODEL_ID
```

## Train a Custom Model

```bash
dm train --model-name mymodel \
  --csv-file path/to/data.csv \
  --smiles-column smiles \
  --value-column target \
  [--classification]
```

- `--classification` enables classification mode.
- Default SMILES column is `smiles`, target column is `target`.

## Run Inference

You can run inference using either a CSV file or direct SMILES input:

```bash
# From a CSV or TXT file
dm predict --model-name mymodel --csv-file inputs.csv

# Direct SMILES
dm predict --model-name mymodel --smiles "CCO"
```

## Co-folding and Affinity Predictions

Explore co-folding capabilities using the following notebooks:

- **[Predict 3D structures](https://github.com/deepmirror/deepmirror-client/blob/main/notebooks/Predict_Structure.ipynb)** of protein–ligand complexes via co-folding [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmirror/deepmirror-client/blob/main/notebooks/Predict_Structure.ipynb)

- **[Apply constraints](https://github.com/deepmirror/deepmirror-client/blob/main/notebooks/Constrained_Predict_Structure.ipynb)** during co-folding to guide the predicted structure [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmirror/deepmirror-client/blob/main/notebooks/Constrained_Predict_Structure.ipynb)

- **[Estimate binding affinity values](https://github.com/deepmirror/deepmirror-client/blob/main/notebooks/Boltz2.ipynb)** alongside structural prediction using Boltz-2 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmirror/deepmirror-client/blob/main/notebooks/Boltz2.ipynb)

## 💡 Tips

- If a token is missing or expired, commands will prompt you to log in again.
- Use `--help` on any command for more details, e.g.:

  ```bash
  dm train --help
  ```
