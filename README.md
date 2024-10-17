# Vietnamese Text Search project

Description: This project aims to create **a Vietnamese benchmark** and **a small LM** for information retrieval task.

## Project structure
- dataset
    - VINLI_reranking
        - construct VINLI_reranking dataset (for evaluating)
        - construct VINLI_triplet dataset (for evaluating)
- retriever
    - evaluation
        - evaluator: contain class to evaluate for each task
        - examples: contain examples to evaluate embedding models on benchmarks
    - examples: contain script to train minilm and phobert from our datasets
    - dataset: method to get tokenize online datasets
    - loss and triloss: contain loss functions used in the papers
    - model: wrapper for embedding models
    - utils: support functions

## Resource
Benchmarks and models are publicly available on Hugging Face. You can explore them [here](https://huggingface.co/ContextSearchLM).

Further specific information will be updated.

## Citation
[^1]: Coming soon