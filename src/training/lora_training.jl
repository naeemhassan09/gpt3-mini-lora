using Revise
using Flux
using Random
using Statistics
using BSON: @save
include("../models/gpt_mini.jl")
include("../data/mnli_preprocessing.jl")
include("../evaluation/cross_validation.jl")

using .MNLIData
using .CrossValidation

cfg = GPTMiniConfig(50, 8, 8, 3)
x_data, y_data, vocab = load_mnli_data(cfg.seq_len, cfg.vocab_size)

accs = run_cross_validation(() -> GPTMini_LoRA(cfg, 2), x_data, y_data, cfg.n_classes; folds=2, batch_size=2)
println("Best accuracy: ", maximum(accs))
@save "best_lora_model.bson" model=GPTMini_LoRA(cfg, 2)