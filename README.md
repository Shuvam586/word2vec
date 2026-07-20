# Efficient Estimation of Word Representations in Vector Space

https://paperswithcode.co/paper/1301.3781

## Introduction

This paper focuses on:
1. learning word embeddings more efficiently from larger corpora of text
2. embeddings should capture both grammatical and semantic relationships between words
3. designing models which give embeddings where linear relationships are preserved:
> Queen = King - Man + Woman

## Dataset

- I have used [text8](https://www.kaggle.com/datasets/gupta24789/text8-word-embedding). download and place the txt file in `/data`.
- use `/scripts/split_dataset.py` to produce `tiny.txt`, `small.txt` and `medium.txt` from `raw.txt`.

## Setup & Usage

1. Install the [dataset](#dataset) and run `/scripts/split_dataset.py`.
2. Train a model using `train_cbow_hs.py` or `train_skipgram_hs.py`. pass command-line args for `epochs`, `size` and `lr`.
3. Explore the model's learnings using `test_similarity.py` and `test_analogy.py`. 

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

## CBOW Model using Hierarchical Softmax

What is hierarchical softmax?
> traditional softmax computes per word probabilities by performing dot products of millions of word vectors by multiplying `W_out` with `h` (mean vector of context words). hierarchical softmax arranges words in a huffman tree and then each internal node has a weight vector, which produces `p`, which is probability of the model choosing to go to the right node. 

This converts a multi-class classification problem into a series of binary classification problems.

## Skipgram Model

very similar intuitively to CBOW architecture. predicts multiple possible words from a single word compared to prediciting a single missing word from multiple context words.

## Random Skipgram Window

> `R ~ Uniform(1, C)`

This is supposed to perform better because words closer to the target are trained more to guess that specific word. Intuitively, words closer to the target contribute significantly more compared to the ones `C-1` words away.

## Quality of Life Improvements:

- `epoch`, `lr` and `corpus_size` arguments can be passed from the terminal; implemented using `argparse`.
- `model_io.py` contains functions to save and load trained models.