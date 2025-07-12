module MNLIData

using Flux, Random, CSV, DataFrames

export load_mnli_data, load_vocab, tokenize_sample

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
function dummy_tokenize(text::String, vocab::Dict{String, Int}, seq_len::Int)
    tokens = split(text)  # simple whitespace tokenizer
    token_ids = [get(vocab, tok, vocab["[UNK]"]) for tok in tokens]

    # Pad or truncate to match seq_len
    if length(token_ids) < seq_len
        pad_id = vocab["[PAD]"]
        token_ids = vcat(token_ids, fill(pad_id, seq_len - length(token_ids)))
    elseif length(token_ids) > seq_len
        token_ids = token_ids[1:seq_len]
    end

    return token_ids
end

function load_mnli_data(seq_len::Int=128, max_samples::Int=1000)
    vocab = load_vocab("src/data/bert-base-uncased/vocab.txt")
    data_path = "MNLI/train.tsv"
    @info "Loading MNLI from: $data_path"

    # Explicitly define required columns and their types
    col_types = Dict(
        "sentence1" => Union{Missing, String},
        "sentence2" => Union{Missing, String},
        "label1"    => Union{Missing, String},
    )

    df = CSV.read(data_path, DataFrame;
        delim='\t',
        ignorerepeated=true,
        select=collect(keys(col_types)),
        types=col_types
    )
    x_data = Vector{Vector{Int}}()
    y_data = Int[]

    n = 0
    for row in eachrow(df)
        if n >= max_samples
            break
        end

        if row.label1 === missing || row.sentence1 === missing || row.sentence2 === missing
            continue
        end

        label_str = strip(row.label1)

        label = label_str == "entailment" ? 1 :
                label_str == "neutral" ? 2 :
                label_str == "contradiction" ? 3 : 0

        if label == 0
            continue
        end

        premise = lowercase(row.sentence1)
        hypothesis = lowercase(row.sentence2)

        p_idx = dummy_tokenize(premise, vocab, seq_len)
        h_idx = dummy_tokenize(hypothesis, vocab, seq_len)
        full_idx = vcat(p_idx, h_idx)
        if length(full_idx) > seq_len
            full_idx = full_idx[1:seq_len]
        elseif length(full_idx) < seq_len
            full_idx = vcat(full_idx, fill(vocab["[PAD]"], seq_len - length(full_idx)))
        end

        push!(x_data, full_idx)
        push!(y_data, label)

        n += 1
    end

    @info "Loaded $(length(x_data)) examples from MNLI"
   
    return x_data, y_data, vocab
end

function tokenize_sample(premise::String, hypothesis::String, vocab::Dict{String, Int}; seq_len::Int=4)
    p_idx = dummy_tokenize(premise, vocab, seq_len)
    h_idx = dummy_tokenize(hypothesis, vocab, seq_len)
    return vcat(p_idx, h_idx)  # concat
end


function test_mnli_debug(data_path::String = "MNLI/train.tsv")
    open("MNLI/train.tsv", "r") do io
        header = readline(io)
        println("Header: ", header)

        for (i, line) in enumerate(eachline(io))
            if i > 5
                break
            end

            parts = split(line, '\t')
            println("Line $i has $(length(parts)) columns")

            if length(parts) >= 17
                println("Premise: ", parts[8])
                println("Hypothesis: ", parts[9])
                println("Label1: ", parts[15])
            else
                println("Bad line: ", line)
            end
        end
    end
end
end  # module