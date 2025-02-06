# Datasets

Format and prepare the datasets with the following file names.

i.e. NLI
```bash
.
└── nli/
    ├── mnli
    │     ├── dev.json
    │     ├── test.json
    │     └── train.json
    └── snli
          ├── dev.json
          ├── test.json
          └── train.json
```

If the datasets are registered in Huggingface Datasets, you can easily use them.

## Sentiment Analysis (SA)
IMDb: https://huggingface.co/datasets/stanfordnlp/imdb 
Yelp: https://huggingface.co/datasets/fancyzhx/yelp_polarity 


## Natural Language Inference (NLI)
MNLI: https://huggingface.co/datasets/nyu-mll/multi_nli 
SNLI: https://huggingface.co/datasets/stanfordnlp/snli 


## Named Entity Recognition (NER)
OntoNotess 5.0: https://catalog.ldc.upenn.edu/LDC2013T19 
