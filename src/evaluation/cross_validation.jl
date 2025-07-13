module CrossValidation

using Flux, Random, Statistics, Optimisers, Dates 
using ..GPTMiniModel: get_lora_params, GPTMiniConfig

export run_cross_validation


function run_cross_validation(
    model_fn, x_data, y_data, n_classes;
    train_step, cfg, epochs=3, folds=10, batch_size=16
)
    N = length(x_data)
    idx = shuffle(1:N)
    fold_size = div(N, folds)
    acc_scores = Float64[]
    total_start = time()

    for i in 1:folds
        @info "========== Fold $i/$folds =========="
        fold_start = time()

        test_idx = idx[((i - 1) * fold_size + 1):min(i * fold_size, N)]
        train_idx = setdiff(idx, test_idx)

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]

        model = model_fn()
        lora_params = get_lora_params(model)
        opt = Optimisers.ADAM(3e-2)
        opt_state = Optimisers.setup(opt, lora_params)

        function make_batches(x, y, bs)
            batches = []
            for i in 1:bs:length(x)
                last_idx = min(i + bs - 1, length(x))
                push!(batches, (x[i:last_idx], y[i:last_idx]))
            end
            return batches
        end

        train_data = make_batches(x_train, y_train, batch_size)

        for epoch in 1:epochs
            epoch_losses = []
            for (x_batch, y_batch) in train_data
                x3d = prepare_batch(x_batch, cfg)
                y_oh = Flux.onehotbatch(y_batch, 1:n_classes)
                loss_val, opt_state = train_step(model, x3d, y_oh, opt_state)
                push!(epoch_losses, loss_val)
            end
            @info "Fold $i, Epoch $epoch, Avg Loss: $(round(mean(epoch_losses), digits=4))"
        end

        # ✅ FIXED: Evaluation logic using findmax
        correct = 0
        invalid_preds = 0

        for (x, y) in zip(x_test, y_test)
            x3d = prepare_batch([x], cfg)
            logits = model(x3d)  # (3, 1)

            logit_vec = vec(logits)
            if length(logit_vec) != n_classes || any(isnan, logit_vec)
                @warn "Invalid logits during evaluation: $logit_vec"
                invalid_preds += 1
                continue
            end

            pred = findmax(logit_vec)[2]  # class index
            correct += (pred == y) ? 1 : 0
        end

        total = length(x_test) - invalid_preds
        acc = total > 0 ? correct / total : NaN
        @info "Fold $i Accuracy: $(isnan(acc) ? "NaN (Invalid predictions)" : round(acc, digits=4))"
        push!(acc_scores, acc)

        fold_duration = time() - fold_start
        @info "Fold $i completed in $(round(fold_duration, digits=2)) seconds"
    end

    total_duration = time() - total_start
    @info "========== Cross-Validation Summary =========="
    for (i, acc) in enumerate(acc_scores)
        acc_str = isnan(acc) ? "NaN" : string(round(acc, digits=4))
        @info "Fold $i Accuracy: $acc_str"
    end

    valid_accs = filter(!isnan, acc_scores)

    if !isempty(valid_accs)
        @info "Best Accuracy   : $(round(maximum(valid_accs), digits=4))"
        @info "Worst Accuracy  : $(round(minimum(valid_accs), digits=4))"
        @info "Average Accuracy: $(round(mean(valid_accs), digits=4))"
        @info "Std Deviation   : $(round(std(valid_accs), digits=4))"
    else
        @warn "No valid accuracies computed. All folds returned NaN."
    end

    @info "Total training time: $(round(total_duration, digits=2)) seconds"
    return acc_scores
end

function prepare_batch(x_batch::Vector{Vector{Int}}, cfg::GPTMiniConfig)
    seq_len = cfg.seq_len
    batch_size = length(x_batch)
    vocab_size = cfg.vocab_size

    # DEBUG: Print the lengths of each sequence in the batch
    for (i, x) in enumerate(x_batch)
        if length(x) != seq_len
            println("[DEBUG] Sequence $i has length $(length(x)) ≠ cfg.seq_len ($seq_len)")
        end
    end

    @assert all(length(x) == seq_len for x in x_batch) "Some inputs do not match cfg.seq_len = $seq_len"

    token_matrix = hcat(x_batch...)  # (seq_len, batch_size)
    flat_tokens = vec(token_matrix)
    x_onehot = Flux.onehotbatch(flat_tokens, 1:vocab_size)
    x_onehot = reshape(x_onehot, vocab_size, seq_len, batch_size)
    x_onehot = permutedims(x_onehot, (2, 3, 1))  # (seq_len, batch_size, vocab_size)
    return Float32.(x_onehot)
end

end # module