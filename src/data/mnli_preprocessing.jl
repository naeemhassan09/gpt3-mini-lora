module MNLIData

using Flux, Random

export load_mnli_data, load_vocab

# ----------------------------------------
# Load vocab.txt into a Dict
# ----------------------------------------
function load_vocab(vocab_path::String)
    vocab = Dict{String, Int}()
    for (i, line) in enumerate(eachline(vocab_path))
        vocab[line] = i
    end
    return vocab
end

# ----------------------------------------
# Tokenizer: simple whitespace-based fallback
# ----------------------------------------
function dummy_tokenize(sentence::String, vocab::Dict{String, Int}, seq_len::Int)
    tokens = split(lowercase(sentence))  # naive whitespace split
    indices = [get(vocab, token, 1) for token in tokens[1:min(end, seq_len)]]
    padded = vcat(indices, fill(1, seq_len - length(indices)))  # pad with [PAD] (assumed index 1)
    return padded
end

# ----------------------------------------
# Load mock MNLI samples
# ----------------------------------------
function load_mnli_data(seq_len::Int=4)
    samples = [
        ("The cat sits", "A feline is sitting", 1),
        ("The sky is blue", "It is night", 3),
        ("He reads a book", "He is reading", 1),
        ("They are running", "They stand still", 3),
        ("Apples are red", "Apples can be green", 2)
    ]

    # Load real vocab file (e.g., ~30k tokens from BERT)
    vocab = load_vocab("src/data/bert-base-uncased/vocab.txt")

    x_data = Vector{Vector{Int}}()
    y_data = Int[]

    for (premise, hypothesis, label) in samples
        p_idx = dummy_tokenize(premise, vocab, seq_len)
        h_idx = dummy_tokenize(hypothesis, vocab, seq_len)
        full_idx = vcat(p_idx, h_idx)  # input length = 2 * seq_len
        push!(x_data, full_idx)
        push!(y_data, label)
    end

    return x_data, y_data, vocab
end

end  # module