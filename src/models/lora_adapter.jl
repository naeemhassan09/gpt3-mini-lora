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

function LoRALinear(base::Dense, r::Int; α::Real = 1.0f0)  # Accept Float64 or Int, convert to Float32
    in_dim, out_dim = size(base.weight)
    A = Dense(in_dim, r, bias=false)
    B = Dense(r, out_dim, bias=false)
    return LoRALinear(base, A, B, Float32(α))  # ensure α is Float32
end

function (l::LoRALinear)(x)
    return l.base(x) .+ (l.α / size(l.B.weight, 1)) * l.B(l.A(x))
end

end # module