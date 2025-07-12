include("../src/models/gpt_mini.jl")

cfg = GPTMiniConfig(vocab_size=50, seq_len=4, d_model=8, n_classes=3)
model = GPTMini(cfg)

println("Parameter count: ", count_parameters(model))  # Should be < 1000