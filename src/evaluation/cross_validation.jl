module CrossValidation

using Flux
using Random
using Statistics
using Optimisers

include("../models/lora_adapter.jl")
using .LoRAAdapter: get_lora_params

export run_cross_validation

function run_cross_validation(model_fn, x_data, y_data, n_classes; epochs=3, folds=2, batch_size=2)
    N = length(x_data)
    idx = shuffle(1:N)
    fold_size = div(N, folds)
    acc_scores = []

    for i in 1:folds
        test_idx = idx[((i-1)*fold_size + 1):min(i*fold_size, N)]
        train_idx = setdiff(idx, test_idx)

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]

        # Create fresh model
        model = model_fn()
        lora_params = get_lora_params(model)
        if isempty(lora_params)
            error("No LoRA parameters found in the model. Check get_lora_params implementation or model structure.")
        end
        opt = Optimisers.ADAM(3e-4)
        opt_state = Optimisers.setup(opt, lora_params)
        loss(x, y) = Flux.crossentropy(model(x), y)

        # Create safe batches without overflow
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
            for (x_batch, y_batch) in train_data
                x3d = cat([reshape(x, :, 1, size(x,2)) for x in x_batch]..., dims=2)
                y_oh = Flux.onehotbatch(y_batch, 1:n_classes)

                gs = Flux.gradient(() -> loss(x3d, y_oh), lora_params)
                opt_state, lora_params = Optimisers.update(opt_state, lora_params, gs)
            end

            avg_loss = mean([
                loss(reshape(x, :, 1, size(x,2)), Flux.onehot(y, 1:n_classes))
                for (x, y) in zip(x_train, y_train)
            ])
            println("Fold $i, Epoch $epoch, Loss: $avg_loss")
        end

        # Evaluation
        correct = 0
        for (x, y) in zip(x_test, y_test)
            x3d = reshape(x, :, 1, size(x,2))
            pred = Flux.onecold(model(x3d), 1:n_classes)
            correct += pred == y ? 1 : 0
        end

        acc = correct / length(test_idx)
        println("Fold $i accuracy: $acc")
        push!(acc_scores, acc)
    end

    println("Average accuracy over $folds folds: ", mean(acc_scores))
    return acc_scores
end

end # module