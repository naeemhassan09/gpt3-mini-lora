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
    acc_scores = []

    for i in 1:folds
        test_idx = idx[((i-1)*fold_size + 1):min(i*fold_size, N)]
        train_idx = setdiff(idx, test_idx)

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]

        model = model_fn()
        lora_params = get_lora_params(model)
        if isempty(lora_params)
            error("No LoRA parameters found in the model. Check get_lora_params implementation or model structure.")
        end
        opt = Optimisers.ADAM(3e-2)
        opt_state = Optimisers.setup(opt, lora_params)
        loss(x, y) = Flux.logitcrossentropy(model(x), y)

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
                x3d = prepare_batch(x_batch, cfg)  # (seq_len, batch_size)
                y_oh = Flux.onehotbatch(y_batch, 1:n_classes)
                loss_val, opt_state = train_step(model, x3d, y_oh, opt_state)
            end

            avg_loss = mean([
            loss(prepare_batch([x], cfg), Flux.onehot(y, 1:n_classes))
            for (x, y) in zip(x_train, y_train)
            ])
            println("Fold $i, Epoch $epoch, Loss: $avg_loss")
        end

        correct = 0
        for (x, y) in zip(x_test, y_test)
            x3d = prepare_batch([x], cfg)  # batch size = 1
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