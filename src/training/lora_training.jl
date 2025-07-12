using Revise, Flux, Random, Statistics, Logging, Optimisers
using BSON: @save
include("../models/gpt_mini.jl")
include("../data/mnli_preprocessing.jl")
include("../evaluation/cross_validation.jl")
using .GPTMiniModel, .MNLIData, .CrossValidation

global_logger(ConsoleLogger(stdout, Logging.Info))

x_data, y_data, vocab = load_mnli_data(64)
cfg = GPTMiniConfig(length(vocab), 128, 64, 3) 
@info typeof(x_data[1]) 
@info "Loaded MNLI data: $(length(x_data)) samples, vocab size: $(length(vocab))"

# Count LoRA-only parameters
test_model = GPTMini_LoRA(cfg, 4)
function count_lora_parameters(model)
    lora_params = get_lora_params(model)
    return sum(length, lora_params)
end
@info "LoRA-only trainable parameters: $(count_lora_parameters(test_model))"
@assert count_lora_parameters(test_model) < 1050 "Model exceeds parameter limit!"

# Define train_step
function train_step(model, x, y, opt_state)
    lora_params = get_lora_params(model)
    function loss_fn(params)
        fwd = forward_with_lora(model, params)
        return Flux.logitcrossentropy(fwd(x), y)
    end
    loss, grads = Flux.withgradient(loss_fn, lora_params)
    @info "Training loss: $loss"
    opt_state, new_params = Optimisers.update(opt_state, lora_params, grads)
    for (p_ref, p_val) in zip(lora_params, new_params)
        p_ref .= p_val
    end
    return loss, opt_state
end

# Run 10-fold CV
opt = Optimisers.ADAM(3e-2)
opt_state = Optimisers.setup(opt, nothing)
accs = run_cross_validation(() -> GPTMini_LoRA(cfg, 8), x_data, y_data, cfg.n_classes;
                            train_step=train_step, cfg=cfg, folds=10, batch_size=16)
@info "Best accuracy: $(maximum(accs))"

# Save parameters
lora_params = get_lora_params(test_model)
@save "lora_params.bson" lora_params