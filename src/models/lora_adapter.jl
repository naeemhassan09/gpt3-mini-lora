# src/models/lora_adapter.jl

using Flux
using Flux: Dense, glorot_uniform
using LinearAlgebra

"LoRA Adapter: Adds low-rank trainable matrices A, B to a frozen Dense layer"
mutable struct LoRALinear
    base::Dense      # Frozen Dense layer
    A::Dense         # Low-rank: rank r, in => r
    B::Dense         # Low-rank: rank r => out
    α::Float32       # Scaling factor
end

function LoRALinear(base::Dense, r::Int, α::Float32=1.0f0)
    A = Dense(size(base.weight, 2), r, initW=glorot_uniform, bias=false)
    B = Dense(r, size(base.weight, 1), initW=zeros, bias=false)  # init as 0
    Flux.freeze!(base)  # Do not train original layer
    return LoRALinear(base, A, B, α)
end

function (l::LoRALinear)(x)
    return l.base(x) .+ l.α .* l.B(l.A(x))
end