using BSON: @load
include("../src/models/gpt_mini.jl")
using .GPTMiniModel: GPTMiniConfig, GPTMini_LoRA, forward_with_lora

# Define config
cfg = GPTMiniConfig(30522, 128, 64, 3)

# Create fresh model
model = GPTMini_LoRA(cfg, 8)

# Load saved LoRA weights
@load "lora_params.bson" lora_params

# Inject LoRA params into model
forward = forward_with_lora(model, lora_params)

# Dummy input (float32 3D array shaped like (seq_len, batch_size, vocab_size))
x_dummy = rand(Float32, cfg.seq_len, 1, cfg.vocab_size)

# Run inference
y_pred = forward(x_dummy)

println("Predicted output (softmax logits): ", y_pred)