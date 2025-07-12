using Revise
using Flux
using Random
using Statistics
using Test
using Pkg

# -------------------------------
# Load Modules
# -------------------------------
include("../src/models/gpt_mini.jl")


# Import from modules
using .GPTMiniModel: GPTMini_LoRA, GPTMini, GPTMiniConfig, count_parameters, MiniSelfAttention, LoRALinear, get_lora_params 


# -------------------------------
# Print Dependencies
# -------------------------------
function print_used_packages()
    println("📦 Packages used in GPTMini model:")
    for (_, pkg) in Pkg.dependencies()
        if hasproperty(pkg, :name) && pkg.name in ["Flux", "Random", "LinearAlgebra", "Statistics", "NNlib"]
            version = isnothing(pkg.version) ? "unknown" : string(pkg.version)
            println("  - $(pkg.name) (Version: $version)")
        end
    end
end

print_used_packages()

# -------------------------------
# Config and Input
# -------------------------------
cfg = GPTMiniConfig(50, 4, 8, 3)  # vocab_size, seq_len, d_model, n_classes
seq_len, batch_size, vocab_size = cfg.seq_len, 2, cfg.vocab_size
x = rand(Float32, seq_len, batch_size, vocab_size)

# -------------------------------
# Regular GPTMini Test
# -------------------------------
println("\n🧪 Testing Regular GPTMini...")
model = GPTMini(cfg)
y = model(x)
println("✅ Output shape: ", size(y))
println("🧮 Parameter count: ", count_parameters(model))

# -------------------------------
# LoRA GPTMini Test
# -------------------------------
println("\n🧪 Testing GPTMini with LoRA...")
lora_model = GPTMini_LoRA(cfg, 2)  # rank r = 2
y_lora = lora_model(x)
println("✅ LoRA Output shape: ", size(y_lora))
println("🧮 LoRA Parameter count: ", count_parameters(lora_model))

# -------------------------------
# Unit Tests
# -------------------------------
@testset "LoRA Model Tests" begin
    cfg = GPTMiniConfig(50, 8, 8, 3)
    model = GPTMini_LoRA(cfg, 2)

    @test model isa GPTMini
    @test model.attn isa GPTMiniModel.MiniSelfAttention
    @test model.attn.Wq isa LoRALinear
    @test model.attn.Wv isa LoRALinear  # Fix type qualification

    x = rand(Float32, cfg.seq_len, 2, cfg.vocab_size)  # Fix shape: (seq_len, batch_size, vocab_size)
    y = model(x)
    @test size(y) == (cfg.n_classes,2 )

    lora_params = get_lora_params(model)
    @test !isempty(lora_params)
    @test all(p -> p isa AbstractArray, lora_params)
end