# src/data/mnli_preprocessing.jl
module MNLIData

using Revise
using Flux
using Random
using Transformers
using Transformers.HuggingFace
using Transformers.TextEncoders: WordPieceModel, encode  # ✅ Fix: import encode

export load_mnli_data

# Load tokenizer from HuggingFace
const encoder = hgf"bert-base-uncased:tokenizer"

# Tokenize a sentence using HuggingFace encoder
function tokenize(sentence::String, vocab::Dict{String, Int}, seq_len::Int)
    tokens = encode(encoder, sentence).token
    indices = [get(vocab, t, 1) for t in tokens[1:min(end, seq_len)]]
    padded = vcat(indices, fill(1, seq_len - length(indices)))
    return padded
end

# Dummy tokenizer (for fallback)
function dummy_tokenize(sentence::String, vocab::Dict{String, Int}, seq_len::Int)
    tokens = split(lowercase(sentence))
    indices = [get(vocab, w, 1) for w in tokens[1:min(end, seq_len)]]
    padded = vcat(indices, fill(1, seq_len - length(indices)))
    return padded
end

function onehot_encode(indices::Vector{Int}, vocab_size::Int)
    return Flux.onehotbatch(indices, 1:vocab_size)
end

function load_mnli_data(seq_len::Int=4, vocab_size::Int=50)
    # Mock samples
    samples = [
        ("The cat sits", "A feline is sitting", 1),
        ("The sky is blue", "It is night", 3),
        ("He reads a book", "He is reading", 1),
        ("They are running", "They stand still", 3),
        ("Apples are red", "Apples can be green", 2)
    ]

    # Build dummy vocabulary
    texts = vcat(getindex.(samples, 1)..., getindex.(samples, 2)...)
    all_words = Set(Iterators.flatten(split.(lowercase.(texts))))
    vocab = Dict{String, Int}(word => i + 1 for (i, word) in enumerate(all_words))
    vocab["<unk>"] = 1  # unknown token index

    x_data = Vector{Any}()
    y_data = Int[]

    for (premise, hypothesis, label) in samples
        p_idx = tokenize(premise, vocab, seq_len)
        h_idx = tokenize(hypothesis, vocab, seq_len)
        full_idx = vcat(p_idx, h_idx)
        x = onehot_encode(full_idx, vocab_size)
        push!(x_data, x)
        push!(y_data, label)
    end

    return x_data, y_data, vocab
end

end  # module