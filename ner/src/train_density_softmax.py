import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from typing import Dict, Optional, Union

import faiss
import numpy as np
import torch
from datasets import load_dataset
from torch import nn
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

from algorithms.density_estimators import MAF, DensityEstimatorWrapper, RealNVPFactory
from models import (
    BaselineDeBERTaV3ForTokenClassification,
    DensitySoftmaxDeBERTaV3ForTokenClassification,
    FaissBruteForceKNNDeBERTaV3ForTokenClassification,
    FaissKNNDeBERTaV3ForTokenClassification,
)
from utils.data_utils import CustomDataCollatorForTokenClassification, prepare_input
from utils.evaluations import evaluation
from utils.schemas import TokenClassifierOutputConf
from utils.tasks import NER
from utils.tokenization_utils import tokenize_and_align_labels_from_json
from utils.train_utils import EarlyStoppingForTransformers

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
    stopping_criteria: str = field(
        default="f1", metadata={"help": "Stopping Criteria for Early Stopping."}
    )
    optimize_type: str = field(
        default="maximize", metadata={"help": "higher is better of lower is better."}
    )
    knn_alpha: float = field(
        default=1.0,
        metadata={
            "help": "weight hyperparameter of knn distance based uncertainty weight."
        },
    )
    knn_topk: int = field(default=50, metadata={"help": "Top-K kNN search."})
    knn_temperature: int = field(
        default=1000,
        metadata={"help": "temperature parameter for kNN distance computation."},
    )
    use_pca: bool = field(
        default=False, metadata={"help": "Whether to use PCA transform."}
    )
    use_recomp: bool = field(
        default=False,
        metadata={
            "help": "Whether to Recomputation for kNN distance weight computation."
        },
    )
    knn_nprobe: int = field(default=8, metadata={"help": "Faiss ANN nprobe."})
    transform_dim: int = field(
        default=256, metadata={"help": "Dimension for Dimensionality Reduction."}
    )
    index_type: str = field(
        default="hnsw", metadata={"help": "Whether to use Faiss index."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_path: str = field(default=None, metadata={"help": "data path"})
    test_data_path: str = field(default=None, metadata={"help": "test data path"})
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
    classifier_lr: float = field(
        default=5e-5, metadata={"help": "learning rate for classifier"}
    )
    calibration_algorithm: str = field(
        default=None,
        metadata={"help": "Choose your calibration algorithm. default is None."},
    )
    tau: float = field(
        default=None,
        metadata={"help": "temperature parameter."},
    )
    run_multiple_seeds: bool = field(
        default=False,
        metadata={"help": "multiple runs with different seeds. seed range is [1, 10]"},
    )
    num_monte_carlo: int = field(
        default=20, metadata={"help": "Number of Monte Carlo approximation samples."}
    )
    spectral_norm_upper_bound: float = field(
        default=1.0, metadata={"help": "spectral normalization upper bound."}
    )
    density_estimator_path: str = field(
        default=None,
        metadata={"help": "Path to density estimator weight and its config."},
    )
    use_regularization: bool = field(
        default=True, metadata={"help": "Whether to divide log prob by instance count"}
    )


def main(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    seed: Optional[int] = None,
):
    encoded_train_samples: np.ndarray = np.load(
        os.path.join(data_args.density_estimator_path, "encoded_samples.npy")
    )
    print(encoded_train_samples.shape)
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

    # prepare model settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    with open(
        os.path.join(data_args.density_estimator_path, "density_estimator_config.json")
    ) as f:
        density_estimator_config = json.load(f)

    if re.search(r"DensitySoftmax", model_args.model_type):
        if re.search(r"RealNVP", density_estimator_config["model_type"]):
            if model_args.use_pca:
                embedding_dim = 256
            else:
                embedding_dim = density_estimator_config["embedding_dim"]
            density_estimator = RealNVPFactory.create(
                embedding_dim=embedding_dim, device=device
            )
            density_estimator.load_state_dict(
                torch.load(
                    os.path.join(
                        data_args.density_estimator_path, "density_estimator.pt"
                    )
                )
            )
        elif re.search(r"MAF", density_estimator_config["model_type"]):
            if model_args.use_pca:
                density_estimator = MAF(dim=256, n_layers=6, hidden_dims=[512])
            else:
                density_estimator = MAF(
                    dim=density_estimator_config["embedding_dim"],
                    n_layers=6,
                    hidden_dims=[512],
                )

            density_estimator.load_state_dict(
                torch.load(
                    os.path.join(
                        data_args.density_estimator_path, "density_estimator.pt"
                    )
                )
            )
        else:
            raise NotImplementedError()

        density_estimator.to(device)
        density_estimator.eval()
        for name, param in density_estimator.named_parameters():
            param.requires_grad = False
        density_estimator_wrapper = DensityEstimatorWrapper(density_estimator)

    if re.search(r"DensitySoftmax", model_args.model_type):
        model = DensitySoftmaxDeBERTaV3ForTokenClassification(
            config=config,
            density_estimator=density_estimator_wrapper,
            max_log_prob=density_estimator_config["max_log_prob"],
            id2labelcount={
                label2id[label]: count
                for label, count in density_estimator_config["label2labelcount"].items()
            },
            feature_normalization=density_estimator_config["feature_normalization"],
            use_regularization=data_args.use_regularization,
            device=device,
            pca_model=faiss.read_VectorTransform(
                os.path.join(data_args.density_estimator_path, "pca.pca")
            )
            if model_args.use_pca
            else None,
        )
        base_model = BaselineDeBERTaV3ForTokenClassification.from_pretrained(
            model_args.model_name_or_path
        )
        model_dict = base_model.state_dict()
        model.load_state_dict(model_dict)
        del base_model

    elif re.search(
        r"FaissBruteForceKNNDeBERTaV3ForTokenClassification", model_args.model_type
    ):
        model = FaissBruteForceKNNDeBERTaV3ForTokenClassification(
            config=config,
            device=device,
            feature_normalization=density_estimator_config["feature_normalization"],
            encoded_train_samples=np.load(
                os.path.join(data_args.density_estimator_path, "encoded_samples.npy")
            ),
            topk=model_args.knn_topk,
            alpha=model_args.knn_alpha,
            use_pca=model_args.use_pca,
            knn_temperature=model_args.knn_temperature,
        )
        base_model = BaselineDeBERTaV3ForTokenClassification.from_pretrained(
            model_args.model_name_or_path
        )
        model_dict = base_model.state_dict()
        model.load_state_dict(model_dict)
        del base_model

    elif re.search(r"FaissKNN", model_args.model_type):
        encoded_train_samples: np.ndarray = np.load(
            os.path.join(data_args.density_estimator_path, "encoded_samples.npy")
        )
        model = FaissKNNDeBERTaV3ForTokenClassification(
            config=config,
            device=device,
            feature_normalization=density_estimator_config["feature_normalization"],
            encoded_train_samples=encoded_train_samples,
            topk=model_args.knn_topk,
            alpha=model_args.knn_alpha,
            use_pca=model_args.use_pca,
            use_recomp=model_args.use_recomp,
            index_type=model_args.index_type,
            knn_temperature=model_args.knn_temperature,
            nprobe=model_args.knn_nprobe,
        )
        base_model = BaselineDeBERTaV3ForTokenClassification.from_pretrained(
            model_args.model_name_or_path
        )
        model_dict = base_model.state_dict()
        model.load_state_dict(model_dict)
        del base_model

    elif re.search(r"Baseline", model_args.model_type):
        module = import_module("models")
        auto_model = getattr(module, model_args.model_type)
        model: PreTrainedModel = auto_model.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
        )

    else:
        raise NotImplementedError()

    # prepare datasets
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

    base_test_dataset = load_dataset(
        "json", data_files={"test": os.path.join(data_args.test_data_path, "test.json")}
    ).shuffle(seed=seed)

    dataset = base_dataset.map(tokenization_partial_func, batched=True)
    test_dataset = base_test_dataset.map(tokenization_partial_func, batched=True)

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
    test_dataset = test_dataset.remove_columns(columns_to_remove)

    # dataset.set_format("torch")
    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset["test"],
        shuffle=True,
        collate_fn=data_collator,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )

    model.to(device)
    model.gradient_checkpointing_enable()
    params_to_update: list[tuple[str, torch.Tensor]] = []
    for name, param in model.named_parameters():
        if re.search(r"density_estimator", name):
            param.requires_grad = False
        else:
            param.requires_grad = True
            params_to_update.append((name, param))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in params_to_update if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p for n, p in params_to_update if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate,
        },
    ]
    optimizer = AdamW(
        params=optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )

    num_train_steps = math.ceil(len(train_dataloader) * training_args.num_train_epochs)
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

    early_stopping = EarlyStoppingForTransformers(
        path=training_args.output_dir,
        patience=5,
        verbose=True,
        optimize_type=model_args.optimize_type,
    )
    best_model: Optional[PreTrainedModel] = None

    for epoch in range(int(training_args.num_train_epochs)):
        batches = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{int(training_args.num_train_epochs)}",
        )
        model.train()
        for batch in batches:
            model.zero_grad()
            batch = prepare_input(batch, device)
            outputs: TokenClassifierOutputConf = model(**batch)

            # gradient clipping
            if hasattr(model, "clip_grad_norm_"):
                model.clip_grad_norm_(training_args.max_grad_norm)
            elif hasattr(optimizer, "clip_grad_norm"):
                optimizer.clip_grad_norm(training_args.max_grad_norm)
            else:
                nn.utils.clip_grad.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=training_args.max_grad_norm
                )

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            batches.set_postfix({"loss": loss.item()})

            model.zero_grad()

        logger.info("Evaluation")
        eval_results = evaluation(
            steps=epoch,
            model=model,
            dataloader=eval_dataloader,
            label_map=label_map,
            output_path=training_args.output_dir,
            calibration_algorithm=data_args.calibration_algorithm,
            device=device,
            writer=None,
            num_monte_carlo=data_args.num_monte_carlo,
            split="dev",
            seed=seed,
        )
        eval_results.update({"learning_rate": lr_scheduler.get_last_lr()[0]})
        is_best = early_stopping(
            eval_results[model_args.stopping_criteria],
            model,
            tokenizer,
            save_flag=False,
        )
        if is_best:
            best_weight = model.state_dict()
            best_model = model.to(device)
            best_model.load_state_dict(best_weight)

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f:
            logger.info("***** Eval results *****")
            for key, value in eval_results.items():
                logger.info("  %s = %s", key, value)
                f.write("%s = %s\n" % (key, value))

        if early_stopping.early_stop:
            logger.info("Early Stopped.")
            break

    # Test
    logger.info("Test")

    # load best model
    # model.to(device)
    test_results = evaluation(
        steps=None,
        model=best_model,
        dataloader=test_dataloader,
        label_map=label_map,
        output_path=training_args.output_dir,
        calibration_algorithm=data_args.calibration_algorithm,
        device=device,
        writer=None,
        num_monte_carlo=data_args.num_monte_carlo,
        split="test",
        seed=seed,
    )

    test_results.update({"learning_rate": lr_scheduler.get_last_lr()[0]})
    output_test_results_file = os.path.join(
        training_args.output_dir, f"test_results_{str(seed)}.txt"
    )
    with open(output_test_results_file, "w") as f:
        logger.info("***** Test results *****")
        for key, value in test_results.items():
            logger.info("  %s = %s", key, value)
            f.write("%s = %s\n" % (key, value))

    # if not Early Stopped, save the final model.
    if early_stopping.early_stop is False:
        # Save Model
        best_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


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

    if data_args.run_multiple_seeds:
        for i in range(1, 11):  # 1 ~ 10
            main(model_args, data_args, training_args, seed=i)
    else:
        main(model_args, data_args, training_args)
