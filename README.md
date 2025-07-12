
# Mini GPT-3 (<1000 params) + LoRA Implementation (MSc AI Project)

This repository contains the coursework for the **MSc Artificial Intelligence - Deep Learning Module**.

## 🎯 Objective

Reproduce the core ideas of the [LoRA paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685) on a toy GPT-3 architecture (with <1000 learnable parameters). We compare:

- **Standard fine-tuning** of all model parameters vs.
- **LoRA (Low-Rank Adaptation)** which adapts the model by injecting low-rank matrices into frozen weights

All experiments are conducted on the **MNLI (Multi-Genre Natural Language Inference)** dataset.

---

## 🗂 Project Structure

```
project_folder/
├── src/
│   ├── models/             # gpt_mini.jl and lora_adapter.jl
│   ├── data/               # mnli_preprocessing.jl
│   ├── training/           # standard_training.jl and lora_training.jl
│   ├── evaluation/         # cross_validation.jl
│   └── utils/              # helpers.jl
├── experiments/            # run_experiments.jl
├── results/                # stores cross-validation output
├── README.md
└── Project.toml            # Julia dependencies
```

---

## ⚡ Quick Start

### 1. Clone & Activate Environment

```bash
git clone https://github.com/naeemhassan09/gpt3-mini-lora.git
cd gpt3-mini-lora
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Preprocess MNLI Dataset

```bash
julia --project=. src/data/mnli_preprocessing.jl
```

To use the full MNLI dataset (~433k samples), make sure you have downloaded it from the GLUE benchmark:
- [MNLI Full Dataset Download](https://nyu-mll.github.io/GLUE/)
- Place the `.tsv` files under `data/mnli_raw/` and ensure paths are correct in `mnli_preprocessing.jl`.

---

## 🧠 Model Training

### Standard Fine-Tuning

```bash
julia --project=. src/training/standard_training.jl
```

### LoRA Training

```bash
julia --project=. src/training/lora_training.jl
```

---

## 🔍 Evaluation (10-Fold Cross-Validation)

```bash
julia --project=. src/evaluation/cross_validation.jl
```

Results (accuracy, F1) will be stored in `results/` and summarized in terminal logs.

---

## 🧪 Using the Trained Model

Example code to run a prediction:

```julia
using Flux, Serialization
model = deserialize("results/lora_model_fold1.bson")
input = "A man inspects the uniform of a figure in some East Asian country."
tokenized = tokenize(input)  # Use your helper function
output = model(tokenized)
```

> Note: You may wrap this into an `inference.jl` script for convenience.

---

## 📊 Results

- Experiments compare **standard vs. LoRA** training.
- Evaluation metrics: **Accuracy, F1-Score** (average over 10 folds)
- Visualizations generated from `results/`

---

## 🧾 References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
- Flux.jl documentation
- MNLI dataset: https://huggingface.co/google-bert/bert-base-uncased)

---

## 📌 Acknowledgments

This project is submitted as part of the MSc Artificial Intelligence coursework at Dublin Business School.
