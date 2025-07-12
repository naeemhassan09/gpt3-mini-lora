using Revise
using Flux
using Random
include("../src/models/gpt_mini.jl")

# Print package versions
print_used_packages()

cfg = GPTMiniConfig(50, 4, 8, 3)
seq_len, batch_size, vocab_size = cfg.seq_len, 2, cfg.vocab_size

# Input shape: (S, B, V)
x = rand(Float32, seq_len, batch_size, vocab_size)

println("✅ Testing Regular GPTMini...")
model = GPTMini(cfg)
y = model(x)
println("Output shape: ", size(y))  # should be (B, n_classes)
println("Parameter count: ", count_parameters(model))