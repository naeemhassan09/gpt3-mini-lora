using Revise, Flux, Random, Statistics, Logging, Optimisers, Zygote, Dates
using BSON: @save
include("../models/gpt_mini.jl")
include("../data/mnli_preprocessing.jl")
include("../evaluation/cross_validation_standard.jl")

using .GPTMiniModel, .MNLIData, .CrossValidationStandard

global_logger(ConsoleLogger(stdout, Logging.Info))

# === CONFIGURATION ===
const MODEL_NAME = "GPTMini_Standard"
const EPOCHS = 3
const BATCH_SIZE = 4
const NUM_FOLDS = 3
const LR = 3e-2
const SEQ_LEN = 16
const EMBEDDING_DIM = 3
const ROWS = 3
const CLASSES = 3

# === LOAD DATA ===
x_data, y_data, vocab = load_mnli_data(SEQ_LEN, ROWS)
@info "Loaded MNLI data: $(length(x_data)) samples, vocab size: $(length(vocab))"
cfg = GPTMiniConfig(length(vocab), SEQ_LEN, EMBEDDING_DIM, CLASSES)

# === INIT MODEL & PARAM COUNT ===
test_model = GPTMini(cfg)
params = Flux.params(test_model)
@info "Standard trainable parameters: $(sum(length, params))"
for (i, p) in enumerate(params)
    @info "Param $i shape: $(size(p)), count: $(length(p))"
end
# @assert sum(length, params) < 1050 "Model exceeds parameter limit!"

# === SETUP OPTIMIZER ===
opt = Optimisers.ADAM(LR)
θ_init, re = Flux.destructure(test_model)
opt_state = Optimisers.setup(opt, θ_init)

# === TRAIN STEP ===
function train_step(model, x, y, opt_state)
    f, re = Flux.destructure(model)
    θ = f

    function loss_fn(θ)
        model_re = re(θ)
        ŷ = model_re(x)
        return Flux.logitcrossentropy(ŷ, y)
    end

    loss, back = Zygote.pullback(loss_fn, θ)
    grads = first(back(1f0))
    @info "Training loss: $loss"

    opt_state, θ_new = Optimisers.update(opt_state, θ, grads)
    updated_model = re(θ_new)
    return loss, opt_state, updated_model
end

# === TRAIN & EVALUATE ===
accs, final_model = run_cross_validation_standard(() -> GPTMini(cfg), x_data, y_data, cfg.n_classes;
                                         train_step=train_step,
                                         cfg=cfg,
                                         folds=NUM_FOLDS,
                                         batch_size=BATCH_SIZE,
                                         epochs=EPOCHS,
                                         opt_state=opt_state)

valid_accs = filter(x -> x !== nothing && !isnan(x), accs)

# === SAVE FINAL TRAINED MODEL ===
@save "standard_model.bson" final_model vocab cfg

# === PRINT SUMMARY ===
@info "========== Training Summary =========="
@info "Model        : $MODEL_NAME"
@info "Seq Length   : $(cfg.seq_len)"
@info "Embedding    : $(cfg.d_model)"
@info "Classes      : $(cfg.n_classes)"
@info "Vocab Size   : $(length(vocab))"
@info "Train Epochs : $EPOCHS"
@info "Batch Size   : $BATCH_SIZE"
@info "Learning Rate: $LR"
@info "Folds        : $NUM_FOLDS"
@info "Trainable Parameters: $(sum(length, Flux.params(final_model)))"

if !isempty(valid_accs)
    @info "Final Best Accuracy   : $(round(maximum(valid_accs), digits=4))"
    @info "Final Avg Accuracy    : $(round(mean(valid_accs), digits=4))"
    @info "Final Std Deviation   : $(round(std(valid_accs), digits=4))"
else
    @warn "All folds returned NaN accuracy. Check logits/prediction logic."
end