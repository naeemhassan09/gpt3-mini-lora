module LoRAAdapter

using Flux
using Flux: @functor

struct LoRALinear
    base::Dense
    A::Dense
    B::Dense
    α::Float32
end

@functor LoRALinear

function LoRALinear(base::Dense, r::Int; α::Real = 1.0f0)
    in_dim, out_dim = size(base.weight)
    A = Dense(in_dim, r, bias=false)
    B = Dense(r, out_dim, bias=false)
    return LoRALinear(base, A, B, Float32(α))
end

function (l::LoRALinear)(x)
    return l.base(x) .+ (l.α / size(l.B.weight, 1)) * l.B(l.A(x))
end

# 

function get_lora_params(model)
    lora_params = []
    function collect_lora_params(m)
        println("Inspecting type: ", typeof(m))
        if m isa LoRAAdapter.LoRALinear
            println("Found LoRALinear layer: ", m)
            append!(lora_params, Flux.params(m.A, m.B))
        elseif m isa Flux.Chain
            println("Found Chain, iterating layers")
            for layer in m
                collect_lora_params(layer)
            end
        elseif m isa Main.GPTMiniModel.LearnablePositionalEncoding
            println("Skipping positional encoding: ", typeof(m))
        elseif m isa Flux.Dense || m isa Flux.LayerNorm
            println("Skipping Dense or LayerNorm: ", typeof(m))
        elseif m isa Main.GPTMiniModel.MiniSelfAttention
            println("Found MiniSelfAttention, checking fields")
            collect_lora_params(m.Wq)
            collect_lora_params(m.Wk)
            collect_lora_params(m.Wv)
            collect_lora_params(m.Wo)
        elseif m isa Main.GPTMiniModel.GPTMini
            println("Found GPTMini, checking fields")
            collect_lora_params(m.embed)
            collect_lora_params(m.pos_enc)
            collect_lora_params(m.attn)
            collect_lora_params(m.ln)
            collect_lora_params(m.classifier)
        else
            println("Unhandled type: ", typeof(m))
        end
    end
    collect_lora_params(model)
    println("LoRA parameters collected: ", length(lora_params))
    return lora_params
end

function assign_lora_params!(model, lora_params)
    # Get the references to the LoRA trainable layers
    model_params = get_lora_params(model)
    for (param_ref, param_val) in zip(model_params, lora_params)
        param_ref .= param_val
    end
end

export  LoRALinear, get_lora_params, assign_lora_params!



end # module