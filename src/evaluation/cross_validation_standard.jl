module CrossValidationStandard

using Flux, Logging, Statistics
import Flux: onehotbatch, onecold

function accuracy(ŷ, y)
    preds = onecold(ŷ)
    truths = onecold(y)
    return mean(preds .== truths)
end

function run_cross_validation_standard(model_fn, x_data, y_data, n_classes;
                                       train_step,
                                       cfg,
                                       epochs=3,
                                       folds=3,
                                       batch_size=4,
                                       opt_state=nothing)

    fold_size = length(x_data) ÷ folds
    accuracies = Float64[]
    trained_model = nothing

    for fold in 1:folds
        @info "========== Fold $fold/$folds =========="
        start_idx = (fold - 1) * fold_size + 1
        end_idx = min(fold * fold_size, length(x_data))

        x_val = x_data[start_idx:end_idx]
        y_val = y_data[start_idx:end_idx]

        x_train = vcat(x_data[1:start_idx-1]..., x_data[end_idx+1:end]...)
        y_train = vcat(y_data[1:start_idx-1]..., y_data[end_idx+1:end]...)

        model = model_fn()

        for epoch in 1:epochs
            @info "Epoch $epoch"
            for i in 1:batch_size:length(x_train)
                x_batch = x_train[i:min(i+batch_size-1, end)]
                y_batch = y_train[i:min(i+batch_size-1, end)]

                # === Safety Check ===
                if isempty(x_batch) || isempty(y_batch)
                    @warn "Skipping empty training batch"
                    continue
                end

                # === Input Tensor ===
                x_tensor = cat([Float32.(onehotbatch(x, 1:cfg.vocab_size)) for x in x_batch]...; dims=3)
                x_tensor = permutedims(x_tensor, (2, 3, 1))  # (seq_len, batch, vocab)

                # === Target Tensor ===
                y_tensor = onehotbatch(y_batch, 1:n_classes)

                # === Final Check ===
                if size(x_tensor, 2) != size(y_tensor, 2)
                    @warn "Batch size mismatch: x=$(size(x_tensor)), y=$(size(y_tensor))"
                    continue
                end

                loss, opt_state, model = train_step(model, x_tensor, y_tensor, opt_state)
            end
        end

        # === Evaluation ===
        if isempty(x_val) || isempty(y_val)
            @warn "Empty validation set for fold $fold"
            continue
        end

        x_val_tensor = cat([Float32.(onehotbatch(x, 1:cfg.vocab_size)) for x in x_val]...; dims=3)
        x_val_tensor = permutedims(x_val_tensor, (2, 3, 1))

        y_val_tensor = onehotbatch(y_val, 1:n_classes)

        preds = model(x_val_tensor)

        if size(preds) != size(y_val_tensor)
            @warn "Validation shape mismatch: preds=$(size(preds)), targets=$(size(y_val_tensor))"
            continue
        end

        acc = accuracy(preds, y_val_tensor)

        if !isnan(acc)
            push!(accuracies, acc)
            @info "Validation accuracy (Fold $fold): $(round(acc, digits=4))"
        else
            @warn "NaN accuracy on fold $fold"
        end

        if fold == 1
            trained_model = model
        end
    end

    # === Summary ===
    @info "========== Cross-Validation Summary =========="
    if !isempty(accuracies)
        @info "Accuracies: $(round.(accuracies, digits=4))"
        @info "Best Fold Accuracy   : $(round(maximum(accuracies), digits=4))"
        @info "Average Accuracy     : $(round(mean(accuracies), digits=4))"
        @info "Std Dev of Accuracy  : $(round(std(accuracies), digits=4))"
    else
        @warn "No valid accuracies recorded."
    end

    return accuracies, trained_model
end

export run_cross_validation_standard
end # module