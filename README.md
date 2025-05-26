# deepmirror

`deepmirror` is a command-line interface for interacting with the [deepmirror API](https://api.app.deepmirror.ai/public/docs). It allows you to train models, run predictions, and submit structure prediction jobs directly from your terminal.

---

## 🚀 Installation

```bash
pip install deepmirror
```

---

## 🔐 Authentication

Before using most commands, you need to log in to get your API token:

```bash
dm login EMAIL
```

This saves your token and host in `~/.config/deepmirror/` for reuse.

---

## 🧠 Model Commands

### 📋 List Available Models

```bash
dm model list
```

### 📄 View Model Metadata

```bash
dm model metadata MODEL_ID
```

### 🔎 Get Full Model Info

```bash
dm model info MODEL_ID
```

---

## 🏋️ Train a Custom Model

```bash
dm train --model-name mymodel \
  --csv-file path/to/data.csv \
  --smiles-column smiles \
  --value-column target \
  [--classification]
```

- `--classification` enables classification mode.
- Default SMILES column is `smiles`, target column is `target`.

---

## 🔮 Run Inference

You can run inference using either a CSV file or direct SMILES input:

```bash
# From a CSV or TXT file
dm predict --model-name mymodel --csv-file inputs.csv

# Direct SMILES
dm predict --model-name mymodel --smiles "CCO"
```

---

## 🧬 Structure Prediction

### 🧠 Predict Protein-Ligand Structure

```bash
dm structure predict protein.pdb ligand.sdf --model chai
```

- Default model is `chai`.
- Protein and ligand must be valid file paths.

### 📦 Download Prediction Result

```bash
dm structure download TASK_ID result.zip
```

### 📃 List Submitted Jobs

```bash
dm structure list
```

---

## ⚙️ Configuration

- API host and token are saved under `~/.config/deepmirror/`.
- You can override the API host:

  - via `--host` option on any command
  - or by setting `DEEPMIRROR_API_ENV=local`

---

## 💡 Tips

- If a token is missing or expired, commands will prompt you to log in again.
- Use `--help` on any command for more details, e.g.:

  ```bash
  dm train --help
  ```
