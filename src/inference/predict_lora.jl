using Flux, Logging, Statistics
using BSON: @load
include("../models/gpt_mini.jl")
include("../data/mnli_preprocessing.jl")  # for tokenize_sample

using .GPTMiniModel
using .MNLIData


global_logger(ConsoleLogger(stdout, Logging.Info))

# Load saved model parameters and config
@load "lora_params.bson" lora_params vocab cfg

# Reconstruct model with correct LoRA rank
lora_rank = 4
model = GPTMini_LoRA(cfg, lora_rank)

# Assign saved LoRA parameters
assign_lora_params!(model, lora_params)

# Example input
premise = "A man inspects the uniform of a figure in some East Asian country."
hypothesis = "The man is sleeping."

x_indices = tokenize_sample(premise, hypothesis, vocab)

if length(x_indices) > cfg.seq_len
    x_indices = x_indices[1:cfg.seq_len]
elseif length(x_indices) < cfg.seq_len
    x_indices = vcat(x_indices, fill(1, cfg.seq_len - length(x_indices)))  # pad with [PAD] token id = 1
end

vocab_size = cfg.vocab_size
seq_len = cfg.seq_len
batch_size = 1

@info "Token indices length: $(length(x_indices))"
@info "Expected seq_len: $seq_len"
@info "Expected vocab_size: $vocab_size"

x_oh = Flux.onehotbatch(x_indices, 1:vocab_size)
@info "One-hot shape (before reshape): $(size(x_oh))"
@info "Total elements: $(length(x_oh))"
@info "Expected total elements after reshape: $(vocab_size * seq_len * batch_size)"

x_tensor = permutedims(Float32.(reshape(Array(x_oh), vocab_size, seq_len, batch_size)), (2, 3, 1))
@info "Input tensor shape: $(size(x_tensor))"

output = model(x_tensor)
pred_class = argmax(output)

labels = ["entailment", "neutral", "contradiction"]
@info "Input Premise: $premise"
@info "Input Hypothesis: $hypothesis"
@info "Predicted class: $(labels[pred_class])"

confidences = vec(output)
@info "Class probabilities:"
for (i, p) in enumerate(confidences)
    @info "  $(labels[i]) => $(round(p * 100, digits=2))%"
end