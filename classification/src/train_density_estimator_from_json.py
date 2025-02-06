import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import Dict, Optional, Union

import faiss
import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from algorithms.density_estimators import RealNVPFactory
from utils.data_utils import prepare_input
from utils.schemas import SequenceClassifierOutputConf
from utils.tokenization_utils import tokenize_and_align_labels_from_json

logger = logging.getLogger(__name__)

MODEL_LIST = [
    "LogitNormalizedBertForTokenClassification",
    "LogitNormalizedBertCRFForTokenClassification",
    "LogitNormalizedDistilBertForTokenClassification",
    "LogitNormalizedElectraForTokenClassification",
    "LogitNormalizedRobertaForTokenClassification",
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_type: str = field(
        default="AutoModelForTokenClassification",
        metadata={"help": f"Model type selected in the list: {','.join(MODEL_LIST)}"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    task_type: Optional[str] = field(
        default="NER",
        metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(default=None, metadata={"help": "data path"})
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    embedding_dim: int = field(default=768, metadata={"help": "embedding size of PLMs"})
    density_estimator_type: str = field(
        default="RealNVP", metadata={"help": "class name of Density estimator."}
    )
    density_estimator_batch_size: int = field(
        default=128, metadata={"help": "batch size for Density estimator."}
    )
    feature_normalization: bool = field(
        default=False, metadata={"help": "Whether to use feature normalization."}
    )
    use_pca: bool = field(default=False, metadata={"help": "Whether to use PCA."})
    pca_embedding_dim: int = field(
        default=256,
        metadata={
            "help": "Embedding dimension for PCA."
        }
    )


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    seed: Optional[int] = None,
):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    os.makedirs(name=training_args.output_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    if seed is None:
        seed = training_args.seed
        set_seed(seed)
    else:
        set_seed(seed)

    # Get datasets
    def get_id2label() -> dict[int, str]:
        return {
            0: "negative",
            1: "positive",
        }

    def get_nli_id2label() -> dict[int, str]:
        return {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
        

    if "nli" in data_args.data_path:
        id2label = get_nli_id2label()
    elif "imdb" in data_args.data_path:
        id2label = get_id2label()
    else:
        raise NotImplementedError()
    
    
    label2id = {l: i for i, l in id2label.items()}
    num_labels = len(id2label)
    padding = "max_length" if data_args.pad_to_max_length else False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model settings
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        cache_dir=model_args.cache_dir,
    )
    tokenizer: Union[
        PreTrainedTokenizer, PreTrainedTokenizerFast
    ] = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    module = import_module("models")
    try:
        auto_model = getattr(module, model_args.model_type)
        model: PreTrainedModel = auto_model.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    except AttributeError:
        raise ValueError(
            f"{model_args.model_type} is not defined."
            f"Available models are: {','.join(MODEL_LIST)}"
        )
    if re.search(r"RealNVP", data_args.density_estimator_type):
        if data_args.use_pca:
            density_estimator = RealNVPFactory.create(
                embedding_dim=data_args.pca_embedding_dim, device=device
            )
        else:
            density_estimator = RealNVPFactory.create(
                embedding_dim=data_args.embedding_dim, device=device
            )

    else:
        raise NotImplementedError()

    base_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.data_path, "train.json"),
            "validation": os.path.join(data_args.data_path, "dev.json"),
        },
    )

    base_dataset = base_dataset.shuffle(seed=seed)
    
    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["text"], padding=padding, max_length=data_args.max_seq_length, truncation=True)
        result["label"] = [l for l in examples["label"]]
        return result
    
    def preprocess_function_nli(examples):
        result = tokenizer(examples["premise"], examples["hypothesis"], padding=padding, max_length=data_args.max_seq_length, truncation=True)
        result["label"] = [l for l in examples["label"]]
        return result

    base_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.data_path, "train.json"),
            "validation": os.path.join(data_args.data_path, "dev.json"),
        },
    )
    
    base_dataset = base_dataset.shuffle(seed=seed)


    if "nli" in data_args.data_path:
        dataset = base_dataset.map(preprocess_function_nli, batched=True)
        dataset = dataset.remove_columns(column_names=["premise", "hypothesis"])
    elif "imdb" in data_args.data_path:
        dataset = base_dataset.map(preprocess_function, batched=True)
        dataset = dataset.remove_columns(column_names=["text"])
    else:
        raise NotImplementedError()

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # dataset.set_format("torch")
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )

    model.to(device)
    density_estimator.to(device)

    if not re.search(r"GMM", data_args.density_estimator_type):
        optimizer_grouped_parameters = [
            p for p in density_estimator.parameters() if p.requires_grad == True
        ]

        optimizer = AdamW(
            params=optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
        )

        num_train_steps = math.ceil(
            len(train_dataloader) * training_args.num_train_epochs
        )
        if training_args.warmup_steps > 0:
            num_warmup_steps = training_args.warmup_steps
        elif training_args.warmup_ratio > 0:
            num_warmup_steps = int(num_train_steps * training_args.warmup_ratio)
        else:
            num_warmup_steps = 0
        num_warmup_steps = 0

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

    embeddings = []
    labels = []
    model.eval()
    batches_tl = tqdm(train_dataloader)

    for batch in batches_tl:
        batch = prepare_input(batch, device)
        with torch.inference_mode():
            outputs: SequenceClassifierOutputConf = model(**batch)
            
        embeddings.append(outputs.sequence_output)
        labels.append(batch["labels"])
            
    all_embeddings = torch.cat(embeddings).to(device)
    all_labels = torch.cat(labels).to(device)

    normalizer = lambda x: x / (
        torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
    )

    if data_args.feature_normalization:
        all_embeddings: torch.Tensor = normalizer(all_embeddings)


    np.save(
        file=os.path.join(training_args.output_dir, "encoded_samples.npy"),
        arr=all_embeddings.detach().cpu().numpy(),
    )

    # PCA
    if data_args.use_pca:
        pca = faiss.PCAMatrix(d_in=data_args.embedding_dim, d_out=data_args.pca_embedding_dim)
        X = all_embeddings.detach().cpu().numpy()
        pca.train(X)
        X_pca = pca.apply(X)

        np.save(
            file=os.path.join(
                training_args.output_dir, "compressed_encoded_samples.npy"
            ),
            arr=X_pca,
        )
        faiss.write_VectorTransform(
            pca, os.path.join(training_args.output_dir, "pca.pca")
        )

        dataset_for_density_estimation = TensorDataset(
            torch.from_numpy(X_pca).to(device)
        )
        dataloader_for_density_estimation = DataLoader(
            dataset_for_density_estimation,
            batch_size=data_args.density_estimator_batch_size,
            shuffle=True,
        )
    else:
        all_embeddings.to(device)
        dataset_for_density_estimation = TensorDataset(all_embeddings)
        dataloader_for_density_estimation = DataLoader(
            dataset_for_density_estimation,
            batch_size=data_args.density_estimator_batch_size,
            shuffle=True,
        )
    max_log_prob = 0
    min_log_prob = 1000000
    for epoch in range(int(training_args.num_train_epochs)):
        batches = tqdm(
            dataloader_for_density_estimation,
            desc=f"Epoch {epoch + 1}/{int(training_args.num_train_epochs)}",
        )
        density_estimator.train()
        for batch in batches:
            inputs = {"x": batch[0].to(device)}
            log_probs = density_estimator.log_prob(**inputs)
            tmp_max_log_prob = float(log_probs.max())
            tmp_min_log_prob = float(log_probs.min())
            if max_log_prob <= tmp_max_log_prob:
                max_log_prob = tmp_max_log_prob

            if min_log_prob >= tmp_min_log_prob:
                min_log_prob = tmp_min_log_prob

            # gradient clipping
            if hasattr(model, "clip_grad_norm_"):
                density_estimator.clip_grad_norm_(training_args.max_grad_norm)
            elif hasattr(optimizer, "clip_grad_norm"):
                optimizer.clip_grad_norm(training_args.max_grad_norm)
            else:
                nn.utils.clip_grad.clip_grad_norm_(
                    parameters=density_estimator.parameters(),
                    max_norm=training_args.max_grad_norm,
                )
            loss = -1 * log_probs.mean()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            batches.set_postfix({"loss": loss.item()})
            density_estimator.zero_grad()

    torch.save(
        density_estimator.state_dict(),
        os.path.join(training_args.output_dir, "density_estimator.pt"),
    )
    with open(
        os.path.join(training_args.output_dir, "density_estimator_config.json"), "w"
    ) as f:
        json.dump(
            {
                "max_log_prob": max_log_prob,
                "min_log_prob": min_log_prob,
                "base_model_name": model.__class__.__name__,
                "model_type": data_args.density_estimator_type,
                "n_epochs": training_args.num_train_epochs,
                "batch_size": data_args.density_estimator_batch_size,
                "embedding_dim": data_args.embedding_dim,
                "feature_normalization": data_args.feature_normalization,
                "use_pca": data_args.use_pca,
                "pca_embedding_dim": data_args.pca_embedding_dim,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    main(model_args, data_args, training_args)
