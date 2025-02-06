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
from torch import distributions, nn
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
    get_linear_schedule_with_warmup,
    set_seed,
)

from algorithms.density_estimators import GMM, MAF, RealNVP, RealNVPFactory
from utils.data_utils import CustomDataCollatorForTokenClassification, prepare_input
from utils.schemas import TokenClassifierOutputConf
from utils.tasks import NER
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
    labels: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."
        },
    )
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
    use_other_under_sampling: bool = field(
        default=False, metadata={"help": "Whether to use undersampling for Other tag"}
    )
    feature_normalization: bool = field(
        default=False, metadata={"help": "Whether to use feature normalization."}
    )
    use_pca: bool = field(default=False, metadata={"help": "Whether to use PCA."})
    pca_embedding_dim: int = field(
        default=256, metadata={"help": "Embedding dimension for PCA."}
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
    token_classification_task = NER()
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    label2id: Dict[str, int] = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    padding = "max_length" if data_args.pad_to_max_length else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model settings
    config: PretrainedConfig = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
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
    elif re.search(r"MAF", data_args.density_estimator_type):
        if data_args.use_pca:
            density_estimator = MAF(
                dim=data_args.pca_embedding_dim,
                n_layers=6,
                hidden_dims=[512],
            )
        else:
            density_estimator = MAF(
                dim=data_args.embedding_dim,
                n_layers=6,
                hidden_dims=[512],
            )
    elif re.search(r"GMM", data_args.density_estimator_type):
        density_estimator = GMM(num_labels=num_labels, gda_size=100)
    else:
        raise NotImplementedError()

    tokenization_partial_func = partial(
        tokenize_and_align_labels_from_json,
        tokenizer=tokenizer,
        padding=padding,
        label2id=label2id,
        max_length=data_args.max_seq_length,
    )
    base_dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_args.data_path, "train.json"),
            "validation": os.path.join(data_args.data_path, "dev.json"),
        },
    )

    base_dataset = base_dataset.shuffle(seed=seed)
    dataset = base_dataset.map(tokenization_partial_func, batched=True)
    # Data collator
    data_collator = CustomDataCollatorForTokenClassification(
        tokenizer,
        id2label=config.id2label,
        label2id=config.label2id,
        max_length=data_args.max_seq_length,
        padding="max_length" if data_args.pad_to_max_length else True,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    columns_to_remove = [
        "id",
    ]
    if re.search(r"tweetner", data_args.data_path):
        columns_to_remove = columns_to_remove + ["date"]
    elif re.search(r"multiconer", data_args.data_path):
        columns_to_remove = [
            "domain",
        ]

    dataset = dataset.remove_columns(columns_to_remove)

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

    token_embeddings = []
    all_labels = []
    model.eval()
    batches_ner = tqdm(train_dataloader)
    num_tokens_each_labels_dict = defaultdict(list)
    if data_args.use_other_under_sampling:
        for batch in batches_ner:
            batch = prepare_input(batch, device)
            with torch.inference_mode():
                outputs: TokenClassifierOutputConf = model(**batch)

            # TODO: Oタグが多いはずなので、balancedにする必要があるかもしれない．
            reshaped_sequence_output = torch.reshape(
                outputs.sequence_output, (-1, outputs.sequence_output.shape[-1])
            )  # [B x L, H]
            reshaped_labels = batch["labels"].flatten()  # [B x L]
            filtered_reshaped_sequence_output = torch.stack(
                [
                    s
                    for s, l in zip(reshaped_sequence_output, reshaped_labels)
                    if l != -100
                ]
            )
            filtered_labels = torch.stack([l for l in reshaped_labels if l != -100])
            all_labels.extend(filtered_labels)
            for so, l in zip(filtered_reshaped_sequence_output, filtered_labels):
                num_tokens_each_labels_dict[int(l)].append(so)

        o_id = label2id["O"]
        second_max_num_tokens = 0
        for tag_name in num_tokens_each_labels_dict.keys():
            print(tag_name, len(num_tokens_each_labels_dict[tag_name]))
            if tag_name == o_id:
                continue
            if len(num_tokens_each_labels_dict[tag_name]) > second_max_num_tokens:
                second_max_num_tokens = len(num_tokens_each_labels_dict[tag_name])

        for tag_name, embeddings in num_tokens_each_labels_dict.items():
            if tag_name == o_id:
                sampled_embeddings = random.sample(embeddings, second_max_num_tokens)
            else:
                sampled_embeddings = embeddings

            token_embeddings.extend(sampled_embeddings)
    else:
        for batch in batches_ner:
            batch = prepare_input(batch, device)
            with torch.inference_mode():
                outputs: TokenClassifierOutputConf = model(**batch)

            # TODO: Oタグが多いはずなので、balancedにする必要があるかもしれない．
            reshaped_sequence_output = torch.reshape(
                outputs.sequence_output, (-1, outputs.sequence_output.shape[-1])
            )  # [B x L, H]
            reshaped_labels = batch["labels"].flatten()  # [B x L]
            filtered_reshaped_sequence_output = torch.stack(
                [
                    s
                    for s, l in zip(reshaped_sequence_output, reshaped_labels)
                    if l != -100
                ]
            )
            filtered_labels = torch.stack([l for l in reshaped_labels if l != -100])
            token_embeddings.extend(filtered_reshaped_sequence_output)
            all_labels.extend(filtered_labels)

    normalizer = lambda x: x / (
        torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
    )
    encoded_samples = torch.stack(token_embeddings)
    if data_args.feature_normalization:
        encoded_samples: torch.Tensor = normalizer(encoded_samples)

    encoded_labels = torch.tensor(all_labels).to(encoded_samples.device)

    id2labelcounter = Counter(encoded_labels.detach().cpu().tolist())
    label2labelcount = {label: 0 for label in labels}
    for l, count in id2labelcounter.most_common():
        label2labelcount[label_map[l]] += count

    """entity_types = set([entity for l in labels if (entity := l[2:])])
    entity_types_counter = {entity: 0 for entity in entity_types}
    for entity_type in entity_types:
        for init in ["B-", "I-"]:
            entity_types_counter[entity_type] += id2labelcounter[
                label2id[init + entity_type]
            ]
    for entity_type, num in entity_types_counter.items():
        for init in ["B-", "I-"]:
            label2labelcount[init + entity_type] = num
    label2labelcount["O"] = id2labelcounter[label2id["O"]]"""

    np.save(
        file=os.path.join(training_args.output_dir, "encoded_samples.npy"),
        arr=encoded_samples.detach().cpu().numpy(),
    )

    # PCA
    if data_args.use_pca:
        pca = faiss.PCAMatrix(
            d_in=data_args.embedding_dim, d_out=data_args.pca_embedding_dim
        )
        X = encoded_samples.detach().cpu().numpy()
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
        encoded_samples.to(device)
        dataset_for_density_estimation = TensorDataset(encoded_samples)
        dataloader_for_density_estimation = DataLoader(
            dataset_for_density_estimation,
            batch_size=data_args.density_estimator_batch_size,
            shuffle=True,
        )
    max_log_prob = 0
    min_log_prob = 1000000
    if re.search(r"GMM", data_args.density_estimator_type):
        pca = PCA(n_components=100)
        X = encoded_samples.detach().cpu().numpy()
        pca.fit(X)
        X_pca = pca.transform(X)
        X_pca = torch.from_numpy(X_pca).to(device)
        density_estimator.fit(
            X=X_pca,
            y=encoded_labels,
            id2label=label_map,
        )
        for d in dataloader_for_density_estimation:
            inputs = {
                "x": torch.from_numpy(pca.transform(d[0].detach().cpu().numpy())).to(
                    device
                )
            }
            log_probs = density_estimator.log_prob(**inputs)
            tmp_max_log_prob = float(log_probs.max())
            tmp_min_log_prob = float(log_probs.min())
            if max_log_prob <= tmp_max_log_prob:
                max_log_prob = tmp_max_log_prob

            if min_log_prob >= tmp_min_log_prob:
                min_log_prob = tmp_min_log_prob

    else:
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
                "label2labelcount": label2labelcount,
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

    """if data_args.run_multiple_seeds:
        for i in range(1, 11):  # 1 ~ 10
            main(model_args, data_args, training_args, seed=i)
    else:
        main(model_args, data_args, training_args)"""
    main(model_args, data_args, training_args)
