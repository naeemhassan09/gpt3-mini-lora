using Revise, Flux, Random, Statistics, Logging, Optimisers
using BSON: @save
include("../models/gpt_mini.jl")
include("../data/mnli_preprocessing.jl")
include("../evaluation/cross_validation.jl")
using .GPTMiniModel, .MNLIData, .CrossValidation

global_logger(ConsoleLogger(stdout, Logging.Info))

# === HARD-CODED EXPERIMENT CONFIG ===
const MODEL_NAME = "GPTMini_LoRA"
const LORA_RANK = 4 # LoRA rank
const EPOCHS = 3 # Number of epochs for training
const BATCH_SIZE = 16 # Batch size for training
const NUM_FOLDS = 10 # Number of folds for cross-validation
const LR = 3e-2 #Learning Rate
const SEQ_LEN = 128
const EMBEDDING_DIM = 64 # Embedding dimension
const ROWS = 10 # Number of Rows from Data file to load
const CLASSES= 3 # Number of output classes
# === Load Data ===
x_data, y_data, vocab = load_mnli_data(SEQ_LEN, ROWS)
@info "Loaded MNLI data: $(length(x_data)) samples, vocab size: $(length(vocab))"
@info typeof(x_data[1]) 

cfg = GPTMiniConfig(length(vocab), SEQ_LEN, EMBEDDING_DIM, CLASSES)

# === Parameter Sanity Check ===
test_model = GPTMini_LoRA(cfg, LORA_RANK)
function count_lora_parameters(model)
    lora_params = get_lora_params(model)
    return sum(length, lora_params)
end
@info "LoRA-only trainable parameters: $(count_lora_parameters(test_model))"
@assert count_lora_parameters(test_model) < 1050 "Model exceeds parameter limit!"

# === Define train_step ===
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

# === Train & Evaluate ===
opt = Optimisers.ADAM(LR)
opt_state = Optimisers.setup(opt, nothing)

@info "Starting training and cross-validation..."
train_time = @elapsed begin
    accs = run_cross_validation(() -> GPTMini_LoRA(cfg, LORA_RANK), x_data, y_data, cfg.n_classes;
                            train_step=train_step, cfg=cfg, folds=NUM_FOLDS, batch_size=BATCH_SIZE)
end
valid_accs = filter(x -> x !== nothing && !isnan(x), accs)

if !isempty(valid_accs)
    @info "Best accuracy: $(maximum(valid_accs))"
    @info "Average accuracy: $(round(mean(valid_accs), digits=4))"
else
    @warn "No valid fold accuracies returned!"
end

# === Save Model Parameters ===
lora_params = get_lora_params(test_model)
@save "lora_params.bson" lora_params vocab cfg

# === Print Summary ===
@info "========== Training Summary =========="
@info "Model       : $MODEL_NAME"
@info "LoRA Rank   : $LORA_RANK"
@info "Seq Length  : $(cfg.seq_len)"
@info "Embedding   : $(cfg.d_model)"
@info "Classes     : $(cfg.n_classes)"
@info "Vocab Size  : $(length(vocab))"
@info "Train Epochs: $EPOCHS"
@info "Batch Size  : $BATCH_SIZE"
@info "Learning Rate: $LR"
@info "Folds       : $NUM_FOLDS"

@info "LoRA Parameters Count: $(sum(length, lora_params))"
for (i, p) in enumerate(lora_params)
    @info "  Param $i size = $(size(p)), mean = $(round(mean(p), digits=4))"
end
@info "Total Training Time (seconds): $(round(train_time, digits=2))"
if !isempty(valid_accs)
    @info "Final Best Accuracy   : $(round(maximum(valid_accs), digits=4))"
    @info "Final Avg Accuracy    : $(round(mean(valid_accs), digits=4))"
    @info "Final Std Deviation   : $(round(std(valid_accs), digits=4))"
else
    @warn "All folds returned NaN accuracy. Check logits/prediction logic."
end