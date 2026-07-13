# Efficient Estimation of Word Representations in Vector Space

https://paperswithcode.co/paper/1301.3781

## Introduction

This paper focuses on:
1. learning word embeddings more efficiently from larger corpora of text
2. embeddings should capture both grammatical and semantic relationships between words
3. designing models which give embeddings where linear relationships are preserved:
> Queen = King - Man + Woman


## Preliminary CBOW Model

Training Loop:

We have `W_in`, `W_out`, `ctx_word_ids`, `target_word_id`

1. **Forward pass**: ctx vectors generated from `W_in[ctx_word_ids]`. h is mean vector. scores obtained by matrix mult'ing `W_out` (V, D) and `h` (D, ) and then softmax'ing.

2. **Loss fn**: scores of all words from vocab are generated in forward pass. ideally, `target_word` should have highest score. hence, loss fn. is `-log(prob[target_word])`. so lower the probability, higher the loss.

3. **Backpropagation**: 
    - so ideal score is one hot representation of target word, so `dscores` basically computes diff between what has been predicated and what is "ideally correct". this is `dscores` (delta scores).
    - `dW_out` computes which vectors cause this version of `dscores`, because we previously did `W_out @ h`.
    - `dh` computes how much context representation should change, as `h` is mean vector.
    - `dcontext` just speads out the delta among all words.
    - update `W_in` and `W_out` accordingly, taking into account `learning_rate`.

## Running Tests

tests can be run with:
```bash
python3 -m tests.<test_name>
```

## k-most similar words

`most_similar(word)` in `similarity.py` returns the k-most similar words to `word`.

## analogy

`analogy()` from `analogy.py` finds `x` where `a : b :: c : x` and `a`, `b` and `c` are known words.