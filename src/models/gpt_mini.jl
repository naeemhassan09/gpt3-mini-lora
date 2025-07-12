#gpt_mini.jl
module GPTMiniModel
using Flux
using Flux: Dense, softmax, LayerNorm, @functor
using Random

include("lora_adapter.jl")
using .LoRAAdapter: LoRALinear, get_lora_params, assign_lora_params!


# ----------------------------------------
# MiniSelfAttention with LoRA
# ----------------------------------------
function MiniSelfAttention_LoRA(d_model::Int, r::Int)
    MiniSelfAttention(
        LoRALinear(Dense(d_model, d_model), r),
        Dense(d_model, d_model),
        LoRALinear(Dense(d_model, d_model), r),
        Dense(d_model, d_model)
    )
end

# ----------------------------------------
# Manual Batched Matrix Multiplication
# ----------------------------------------
function manual_batched_mul(A::Array{Float32,3}, B::Array{Float32,3})
    @assert size(A, 1) == size(B, 1) "Batch size mismatch!"
    @assert size(A, 3) == size(B, 2) "Inner dim mismatch!"

    BATCH = size(A, 1)
    # Compute each batched matmul separately and collect into array
    C_list = [A[b, :, :] * B[b, :, :] for b in 1:BATCH]  # List of (M×N) matrices
    return permutedims(cat(C_list..., dims=3), (3, 1, 2))  # Shape: (BATCH, M, N)
end

# ----------------------------------------
# Config
# ----------------------------------------
struct GPTMiniConfig
    vocab_size::Int
    seq_len::Int
    d_model::Int
    n_classes::Int
end

# ----------------------------------------
# Positional Encoding
# ----------------------------------------
struct LearnablePositionalEncoding
    P::Matrix{Float32}
end
@functor LearnablePositionalEncoding

function LearnablePositionalEncoding(seq_len, d_model)
    P = randn(Float32, seq_len, d_model)
    LearnablePositionalEncoding(P)
end

function (pe::LearnablePositionalEncoding)(x)
    return x .+ reshape(pe.P, size(pe.P,1), 1, size(pe.P,2))
end

# ----------------------------------------
# Apply dense layer to (S, B, V) or (S, B, D) input
# ----------------------------------------
function apply_dense3d(d, x::Array{Float32, 3})
    S, B, in_dim = size(x)
    x_flat = permutedims(x, (3, 1, 2))         # (in_dim, S, B)
    x_flat = reshape(x_flat, in_dim, S * B)    # (in_dim, S*B)
    y_flat = d(x_flat)                         # (out_dim, S*B)

    out_dim = size(Flux.params(d)[1], 1)
    y = reshape(y_flat, out_dim, S, B)         # (out_dim, S, B)
    return permutedims(y, (2, 3, 1))           # (S, B, out_dim)
end

# ----------------------------------------
# Self-Attention
# ----------------------------------------
struct MiniSelfAttention
    Wq
    Wk
    Wv
    Wo
end
@functor MiniSelfAttention

function MiniSelfAttention(d_model)
    MiniSelfAttention(
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
        Dense(d_model, d_model),
    )
end

function (m::MiniSelfAttention)(x)
    Q = apply_dense3d(m.Wq, x)
    K = apply_dense3d(m.Wk, x)
    V = apply_dense3d(m.Wv, x)

    Q_perm = permutedims(Q, (2, 1, 3))  # (B, S, D)
    K_perm = permutedims(K, (2, 3, 1))  # (B, D, S)
    attn_scores = manual_batched_mul(Q_perm, K_perm) ./ sqrt(Float32(size(Q, 3)))  # (B, S, S)
    attn_weights = softmax(attn_scores, dims=3)  # (B, S, S)

    V_perm = permutedims(V, (2, 1, 3))  # (B, S, D)
    context = manual_batched_mul(attn_weights, V_perm)  # (B, S, D)
    context = permutedims(context, (2, 1, 3))  # (S, B, D)

    output = apply_dense3d(m.Wo, context)  # (S, B, D)

    @info "[MiniSelfAttention]" x_size=size(x) Q_size=size(Q) K_size=size(K) V_size=size(V) attn_scores=size(attn_scores) context_size=size(context) output_size=size(output)

    return output
end

# ----------------------------------------
# GPTMini Model
# ----------------------------------------
struct GPTMini
    embed::Dense
    pos_enc::LearnablePositionalEncoding
    attn::MiniSelfAttention
    ln::LayerNorm
    classifier::Dense
end
@functor GPTMini

function GPTMini(cfg::GPTMiniConfig)
    embed = Dense(cfg.vocab_size, cfg.d_model)
    pos = LearnablePositionalEncoding(cfg.seq_len, cfg.d_model)
    attn = MiniSelfAttention(cfg.d_model)
    ln = LayerNorm(cfg.d_model)
    classifier = Dense(cfg.seq_len * cfg.d_model, cfg.n_classes)
    return GPTMini(embed, pos, attn, ln, classifier)
end

function (m::GPTMini)(x::Array{Float32, 3})
    h = apply_dense3d(m.embed, x)     # (S, B, D)
    h = m.pos_enc(h)                  # (S, B, D)
    h = m.attn(h)                     # (S, B, D)
    h = permutedims(h, (3, 1, 2))     # (D, S, B)
    h = m.ln(h)                       # (D, S, B)
    h = permutedims(h, (2, 3, 1))     # (S, B, D)
    h_flat = reshape(h, :, size(h, 2))  # (S*D, B)
    y = m.classifier(h_flat)           # (n_classes, B)
    y_out = softmax(y)                 # (n_classes, B)

    @info "[GPTMini Forward]" input_size=size(x) after_embed=size(h) after_posenc=size(h) after_attn=size(h) after_ln=size(h) after_reshape=size(h_flat) logits=size(y) output=size(y_out)

    return y_out
end

function count_parameters(model)
    return sum(length, Flux.params(model))
end

# ----------------------------------------
# GPTMini with LoRA
# ----------------------------------------
function GPTMini_LoRA(cfg::GPTMiniConfig, r::Int)
    embed = Dense(cfg.vocab_size, cfg.d_model)
    pos = LearnablePositionalEncoding(cfg.seq_len, cfg.d_model)
    attn = MiniSelfAttention_LoRA(cfg.d_model, r)
    ln = LayerNorm(cfg.d_model)
    classifier = Dense(cfg.seq_len * cfg.d_model, cfg.n_classes)
    return GPTMini(embed, pos, attn, ln, classifier)
end

function MiniSelfAttention_LoRA(d_model::Int, r::Int)
    MiniSelfAttention(
        LoRALinear(Dense(d_model, d_model), r),
        Dense(d_model, d_model),
        LoRALinear(Dense(d_model, d_model), r),
        Dense(d_model, d_model)
    )
end

function forward_with_lora(model, lora_params::Vector)
    # Shallow copy of model
    model_copy = deepcopy(model)

    # Inject LoRA parameters
    local idx = 1
    for layer in Flux.params(model_copy)
        if layer isa GPTMiniModel.LoRALinear
            layer.A.weight .= lora_params[idx]; idx += 1
            layer.B.weight .= lora_params[idx]; idx += 1
        end
    end

    return x -> model_copy(x)
end

export GPTMini, GPTMini_LoRA, GPTMiniConfig, count_parameters, MiniSelfAttention_LoRA, LoRALinear , get_lora_params, forward_with_lora, assign_lora_params!

end