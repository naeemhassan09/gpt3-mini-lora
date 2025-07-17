
# 🧠 Mini GPT-3 + LoRA in Julia (<1000 Parameters)  
**MSc AI – Deep Learning (B9AI104) Project**  
Author: Naeem ul Hassan •  
📅 Date: July 13, 2025

---

## 🎯 Project Objective

This project reproduces a scaled-down version of GPT-3 (termed **GPTMini**) and applies **LoRA (Low-Rank Adaptation)** to demonstrate parameter-efficient fine-tuning in **Julia** using the `Flux.jl` library.

We compare:
- ✅ **Standard fine-tuning** with ~91K parameters  
- ✅ **LoRA fine-tuning** with only **~1K trainable parameters**

All experiments are conducted on the **MNLI (Multi-Genre Natural Language Inference)** dataset using stratified k-fold cross-validation.

🔗 [LoRA Paper (Hu et al., 2022)](https://arxiv.org/abs/2106.09685)

---

## 📁 Project Structure

```
gpt3-mini-lora/
├── src/
│   ├── models/             # gpt_mini.jl, lora_adapter.jl
│   ├── data/               # mnli_preprocessing.jl
│   ├── training/           # standard_training.jl, lora_training.jl
│   ├── evaluation/         # cross_validation.jl
│   └── utils/              # helpers.jl (optional utilities)
├── experiments/            # run_experiments.jl
├── results/                # output of cross-validation
├── README.md
└── Project.toml            # Julia dependencies
```

---

## ⚡ Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/naeemhassan09/gpt3-mini-lora.git
cd gpt3-mini-lora
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Prepare the MNLI Dataset

- Download the MNLI dataset from the [https://huggingface.co/google-bert/bert-base-uncased)
- Place `train.tsv` into `MNLI/` directory
- Run preprocessing:
- Datadownload link:  wget https://dl.fbaipublicfiles.com/glue/data/MNLI.zip
```bash
julia --project=. src/data/mnli_preprocessing.jl
```

---

## 🚀 Training Models

### 🟩 Standard Fine-Tuning

```bash
julia --project=. src/training/standard_training.jl
```

- Trainable parameters: ~91,818  
- Uses 3-fold CV, `d_model=3`, `seq_len=16`

### 🟦 LoRA Fine-Tuning

```bash
julia --project=. src/training/lora_training.jl
```

- Trainable parameters: **1,024 only**  
- Uses 10-fold CV, `d_model=64`, `seq_len=128`, `LoRA rank = 4`

---

## 📊 Evaluation

Both training scripts run cross-validation internally and print:

- Accuracy per fold  
- Average and Std deviation  
- Total training time  
- LoRA layer statistics (if applicable)

---

## 🧪 Inference Example

```julia
using BSON, Flux
model_data = BSON.load("lora_params.bson")
model = GPTMiniModel.GPTMini_LoRA(model_data[:cfg], 4)
GPTMiniModel.assign_lora_params!(model, model_data[:lora_params])
```

---

## 📈 Results Summary

| Metric             | Standard GPTMini | LoRA GPTMini |
|--------------------|------------------|---------------|
| Avg Accuracy       | 0.303            | 0.300         |
| Best Fold Accuracy | 0.3636           | 1.000         |
| Std Deviation      | 0.0606           | 0.483         |
| Params Trained     | 91,818           | 1,024         |
| Train Time         | ~85s             | ~97s          |

> LoRA matched average performance of full fine-tuning with 90× fewer parameters, validating the findings of Hu et al. (2022). However, it showed higher variance, possibly due to initialization or regularization issues in small-scale settings.

---

## 🧾 References

- Hu, E., Shen, Y., Wallis, P., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)  
- Brown, T. B., et al. (2020). *Language Models are Few-Shot Learners*. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)  
- Flux.jl – [https://fluxml.ai](https://fluxml.ai)

---

## 📌 Acknowledgments

This project was submitted as part of the MSc Artificial Intelligence coursework at **Dublin Business School**, under the module **B9AI104 – Deep Learning**.

✍️ Author: **Naeem ul Hassan**  
📧 naeemhassan09@gmail.com
📅 Submitted: **13 July 2025**
