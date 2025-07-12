module GPTMiniModel
using Flux
using Flux: Dense, softmax, LayerNorm, @functor
using Random

include("lora_adapter.jl")
using .LoRAAdapter: LoRALinear, get_lora_params


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

    BATCH, M, K = size(A)
    _, _, N = size(B)
    C = Array{Float32}(undef, BATCH, M, N)

    for b in 1:BATCH
        C[b, :, :] = A[b, :, :] * B[b, :, :]
    end

    return C
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

    println("Q shape: ", size(Q))
    println("K shape: ", size(K))
    println("V shape: ", size(V))

    Q_perm = permutedims(Q, (2, 1, 3))  # (B, S, D)
    K_perm = permutedims(K, (2, 3, 1))  # (B, D, S)
    println("Q_perm shape: ", size(Q_perm))
    println("K_perm shape: ", size(K_perm))
    attn_scores = manual_batched_mul(Q_perm, K_perm) ./ sqrt(Float32(size(Q, 3)))  # (B, S, S)
    println("attn_scores shape: ", size(attn_scores))

    attn_weights = softmax(attn_scores, dims=3)  # (B, S, S)
    println("attn_weights shape: ", size(attn_weights))

    V_perm = permutedims(V, (2, 1, 3))  # (B, S, D)
    println("V_perm shape: ", size(V_perm))
    context = manual_batched_mul(attn_weights, V_perm)  # (B, S, D)
    println("context shape: ", size(context))
    context = permutedims(context, (2, 1, 3))  # (S, B, D)
    println("context permuted shape: ", size(context))

    output = apply_dense3d(m.Wo, context)  # (S, B, D)
    println("output shape: ", size(output))
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
    println("Input x shape: ", size(x))
    h = apply_dense3d(m.embed, x)     # (S, B, D)
    println("After embed shape: ", size(h))
    h = m.pos_enc(h)                  # (S, B, D)
    println("After pos_enc shape: ", size(h))
    h = m.attn(h)                     # (S, B, D)
    println("After attn shape: ", size(h))
    h = permutedims(h, (3, 1, 2))     # (D, S, B)
    println("Before LayerNorm shape: ", size(h))
    h = m.ln(h)                       # (D, S, B)
    println("After LayerNorm shape: ", size(h))
    h = permutedims(h, (2, 3, 1))     # Back to (S, B, D)
    println("After permute back shape: ", size(h))
    h_flat = reshape(h, :, size(h, 2))  # (S*D, B)
    println("After reshape shape: ", size(h_flat))
    y = m.classifier(h_flat)           # (n_classes, B)
    println("After classifier shape: ", size(y))
    return softmax(y')                 # (B, n_classes)
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

export GPTMini, GPTMini_LoRA, GPTMiniConfig, count_parameters, MiniSelfAttention_LoRA, LoRALinear , get_lora_params 

end