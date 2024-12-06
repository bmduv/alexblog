---
title: NLP
draft: true
tags:
---
# Word Embedding

### One-hot Encoding:
One popular method for word embedding is one-hot encoding of the words. For a list of words of size N, we would create a vector of size N that corresponds to a different integer from 0 to N-1 for each unique word in the data. However this method is not a good choice as this one-hot encoding cannot express the similarity between words, specifically, they cannot be used for cosine similarity for vectors since the cosine similarity between one-hot encoded vectors will always be 0.

### Self-supervised Word2Vec:
This tool is better able to embed words by mapping each word to a fixed-length vector that is better able to express the similarity and relationship between words. Mainly using two models (skip-gram and continuous bag of words (CBOW)), the training uses conditional probabilities that uses the surrounding words to predict the output word.