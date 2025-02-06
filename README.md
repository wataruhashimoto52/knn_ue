# Efficient Nearest Neighbor based Uncertainty Estimation for Natural Language Processing Tasks
Code for "Efficient Nearest Neighbor based Uncertainty Estimation for Natural Language Processing Tasks" in Findings of NAACL 2025.


## Directory Structure

```
.
├── classification/  # classification tasks (sentiment analysis and natural language inference)
└── ner/  # named entity recognition
```

## Usages

### Build environments
We used Singularity environment.

```bash
module load singularity
singularity build --fakeroot research-dev.sif research-dev.def
```

### Prepare datasets

Please see `data/README.md`.

### Run for classification Tasks

```bash
cd classification
sbatch batch_run_custom_classification.sh
```


### Run for NER

```bash
cd ner
sbatch batch_run_custom_ner.sh
```
