
import logging
import math
import json
import os
import re
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Optional, Union

import faiss
import torch
import numpy as np
import datasets
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
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)
from algorithms.density_estimators import (
    DensityEstimatorWrapper,
    RealNVPFactory,
)
from models import (
    BaselineDeBERTaV3ForSequenceClassification,
    DensitySoftmaxDeBERTaV3ForSequenceClassification,
    AutoTuneFaissKNNDeBERTaV3ForSequenceClassification,
    SNGPDeBERTaV3ForSequenceClassification,
    AutoTuneDACDeBERTaV3ForSequenceClassification,
    MDSNDeBERTaV3ForSequenceClassification,
    PosteriorNetworksDeBERTaV3ForSequenceClassification,
    NUQDeBERTaV3ForSequenceClassification,
)
from utils.data_utils import prepare_input, DataCollatorWithPaddingForOOD
from utils.evaluations import evaluation
from utils.schemas import SequenceClassifierOutputConf
from utils.train_utils import EarlyStoppingForTransformers

logger = logging.getLogger(__name__)

MODEL_LIST = [
    "LogitNormalizedBertForTokenClassification",
    "LogitNormalizedBertCRFForTokenClassification",
    "LogitNormalizedDistilBertForTokenClassification",
    "LogitNormalizedElectraForTokenClassification",
    "LogitNormalizedRobertaForTokenClassification",
]

def generate_ood_datasets(id_dataset: str) -> list[str]:
    """fdu_mtl_set = set({
        "apparel",
        "baby",
        "books",
        "dvd",
        "electronics",
        "health_personal_care",
        "imdb",
        "kitchen_housewares",
        "magazines",
        "MR",
        "music",
        "software",
        "sports_outdoors",
        "toys_games",
        "video",
    })"""
    
    other_set = ["imdb", "yelp_polarity"]
    nli_other_set = ["mnli", "snli"]
    if id_dataset in other_set:
        return other_set
    elif id_dataset in nli_other_set:
        return nli_other_set
    else:
        raise NotImplementedError()


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
    use_ood_test: bool = field(
        default=False,
        metadata={"help": "use in-domain test data or out-of-domain test datasets."},
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
    use_feature_normalization: bool = field(
        default=False,
        metadata={
            "help": "Whether to use feature normalization."
        }
    )
    use_dac_only_last_layer: bool = field(
        default=False,
        metadata={
            "help": "whether to use only last layer in DAC algorithm."
        }
    )
    knn_nprobe: int = field(default=8, metadata={"help": "Faiss ANN nprobe."})
    transform_dim: int = field(
        default=256,
        metadata={
            "help": "Dimension for Dimensionality Reduction."
        }
    )
    index_type: str = field(
        default="FLAT",
        metadata={
            "help": "Which to use Faiss index."
        }
    )
    n_list: int = field(
        default=100,
        metadata={
            "help": "Number of clusters for vector pruning."
        }
    )
    n_subvectors: int = field(
        default=64,
        metadata={
            "help": "Number of sub-vectors for product quantization."
        }
    )
    use_neighbor_labels: bool = field(
        default=False,
        metadata={
            "help": "Whether to use neighbor labels in kNN Softmax scoring"
        }
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
    belief_matching_coeff: float = field(
        default=1e-2, metadata={"help": "Coefficient to KL term in BM loss."}
    )
    belief_matching_prior: float = field(
        default=1.0, metadata={"help": "Dirichlet prior parameter."}
    )
    save_model: bool = field(
        default=False, metadata={"help": "whether to save the model."}
    )
    smoothing: float = field(
        default=1.0, metadata={
            "help": "Label Smoothing parameter."
        }
    )
    density_estimator_path: str = field(
        default=None,
        metadata={"help": "Path to density estimator weight and its config."},
    )
    pn_density_type: str = field(
        default="radial_flow",
        metadata={
            "help": "Density estimator type for Posterior Networks"
        }
    )
    pn_latent_dim: int = field(
        default=10,
        metadata={
            "help": "Latent Dimension for Posterior Networks"
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
        
    dataset2labelcount = {
        "imdb": [
            12500,
            12500,
        ],
        "mnli": [
            130899,
            130900,
            130903,
        ]
    }
        

    if "nli" in data_args.data_path:
        id2label = get_nli_id2label()
        label2count = dataset2labelcount["mnli"]
    elif "imdb" in data_args.data_path:
        id2label = get_id2label()
        label2count = dataset2labelcount["imdb"]
    else:
        raise NotImplementedError()
    
    
    label2id = {l: i for i, l in id2label.items()}
    num_labels = len(id2label)
    padding = "max_length" if data_args.pad_to_max_length else False

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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    module = import_module("models")
    

    try:
        auto_model = getattr(module, model_args.model_type)
        if re.search(r"(LogitNormalized|TemperatureScaled)", model_args.model_type):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                device=device,
                tau=data_args.tau,
            )
            
        elif re.search(r"LabelSmoothing", model_args.model_type):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                smoothing=data_args.smoothing,
            )
            
        elif re.search(r"MDSN", model_args.model_type):
            model: PreTrainedModel = MDSNDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                config=config,
            )
            
        elif re.search(r"NUQ", model_args.model_type):
            model: PreTrainedModel = NUQDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                config=config,
            )
        
        elif re.search(
            r"(LastLayerSVI|LastLayerLaplace|BatchEnsemble)", model_args.model_type
        ):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                num_monte_carlo=data_args.num_monte_carlo,
            )
            if re.search(r"FullLayerBatchEnsemble", model_args.model_type):
                model.replace_linear_to_batchensemble_linear()

        elif re.search(
            r"BeliefMatching",
            model_args.model_type,
        ):
            model: PreTrainedModel = auto_model.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                coeff=data_args.belief_matching_coeff,
                prior=data_args.belief_matching_prior,
            )
            
        elif re.search(r"DensitySoftmax", model_args.model_type):
            with open(
                os.path.join(data_args.density_estimator_path, "density_estimator_config.json")
            ) as f:
                density_estimator_config = json.load(f)
        
            if model_args.use_pca:
                embedding_dim = 256
            else:
                embedding_dim = density_estimator_config["embedding_dim"]
            
            density_estimator = RealNVPFactory.create(embedding_dim=embedding_dim, device=device)
            density_estimator.load_state_dict(
                torch.load(
                    os.path.join(
                        data_args.density_estimator_path, "density_estimator.pt"
                    )
                )
            )
            density_estimator.to(device)
            density_estimator.eval()
            for name, param in density_estimator.named_parameters():
                param.requires_grad = False
            density_estimator_wrapper = DensityEstimatorWrapper(density_estimator)
            
            model = DensitySoftmaxDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                config=config,
                density_estimator=density_estimator_wrapper,
                max_log_prob=density_estimator_config["max_log_prob"],
                feature_normalization=model_args.use_feature_normalization,
                device=device,
                pca_model=faiss.read_VectorTransform(
                    os.path.join(data_args.density_estimator_path, "pca.pca")
                )
                if model_args.use_pca
                else None,
            )
            
        elif re.search(r"PosteriorNetworks", model_args.model_type):
            model = PosteriorNetworksDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                config=config,
                label2count=label2count,
                feature_normalization=model_args.use_feature_normalization,
                device=device,
                budget_function="id",
                density_type=data_args.pn_density_type,
                latent_dim=data_args.pn_latent_dim,
            )
            
            
        elif re.match(r"AutoTuneFaissKNNDeBERTaV3ForSequenceClassification", model_args.model_type):
            encoded_train_samples: np.ndarray = np.load(
                os.path.join(data_args.density_estimator_path, "encoded_samples.npy")
            )
            
            model = AutoTuneFaissKNNDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                device=device,
                feature_normalization=model_args.use_feature_normalization,
                encoded_train_samples=encoded_train_samples,
                topk=model_args.knn_topk,
                alpha=model_args.knn_alpha,
                use_pca=model_args.use_pca,
                use_recomp=model_args.use_recomp,
                index_type=model_args.index_type,
                knn_temperature=model_args.knn_temperature,
                nprobe=model_args.knn_nprobe,
                n_list=model_args.n_list,
                n_subvectors=model_args.n_subvectors,
                use_neighbor_labels=model_args.use_neighbor_labels,
            )
            

        elif re.search(r"AutoTuneDAC", model_args.model_type):
            model = AutoTuneDACDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                cache_dir=model_args.cache_dir,
                config=config,
                device=device,
                feature_normalization=model_args.use_feature_normalization,
                use_only_last_layer=model_args.use_dac_only_last_layer,
                topk=model_args.knn_topk,
                use_pca=model_args.use_pca,
                use_recomp=model_args.use_recomp,
                index_type=model_args.index_type,
                nprobe=model_args.knn_nprobe,
            )
            
        elif re.search(
                r"SNGP",
                model_args.model_type,
            ):
            model = SNGPDeBERTaV3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                device=device,
            )

        else:
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
        padding="longest",
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
    eval_dataloader = torch.utils.data.DataLoader(
        dataset["validation"],
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
    
    """optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in params_to_update
                if not any(nd in n for nd in no_decay) and "density_estimation" not in n
            ],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p
                for n, p in params_to_update if "density_estimation" in n
            ],
            "weight_decay": training_args.weight_decay,
            "lr": 5e-4,
        },
        {
            "params": [
                p
                for n, p in params_to_update
                if "density_estimation" not in n and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": training_args.learning_rate,
        },
    ]"""
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in params_to_update
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": training_args.weight_decay,
            "lr": training_args.learning_rate,
        },
        {
            "params": [
                p
                for n, p in params_to_update
                if any(nd in n for nd in no_decay)
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
    
    # best_model = model
    best_model: Optional[PreTrainedModel] = model
    
    for epoch in range(int(training_args.num_train_epochs)):
        batches = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}/{int(training_args.num_train_epochs)}",
        )
        model.train()
        for batch in batches:
            model.zero_grad()
            batch = prepare_input(batch, device)
            outputs: SequenceClassifierOutputConf = model(**batch)

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
            label_map=id2label,
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
            save_flag=data_args.save_model,
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
    if re.search(r"TemperatureScaled", model_args.model_type):
        temperature = best_model.get_optimized_temperature(eval_dataloader)
        best_model.set_is_test(True)
        
    elif re.search(r"(MDSN|NUQ)", model_args.model_type):
        temperature = best_model.fit_ue(eval_dataloader)
        best_model.set_is_test(True)
        
    elif re.search(r"(AutoTuneFaissKNN|AutoTuneLabelRejection)", model_args.model_type):
        best_model.build_index(train_dataloader)
        parameters = best_model.optimize_knn_parameters_numpy(eval_dataloader)
        # best_model.set_knn_temperature(parameters.temperature)
        best_model.set_is_test(True)
        
    elif re.search(r"AutoTuneDAC", model_args.model_type):
        logger.info("DAC embeddings are building...")
        best_model.build_layer_wise_embeddings(train_dataloader)
        logger.info("DAC parameters optimization starts.")
        parameters: list[float] = best_model.optimize_dac_parameters_numpy(eval_dataloader)
        best_model.set_is_test(True)
        
    elif re.search(r"SNGP", model_args.model_type):
        best_model.set_is_test(True)
        
    splitted_path = data_args.data_path.split("/")
    train_dataset_name = splitted_path[-1]
    if model_args.use_ood_test:
        test_dataset_names = generate_ood_datasets(train_dataset_name)
    else:
        test_dataset_names = [train_dataset_name]
    
    test_datasets: list[datasets.Dataset] = []
    for test_dataset_name in test_dataset_names:
        base_test_dataset = load_dataset(
            "json", data_files={"test": os.path.join(data_args.data_path.replace(train_dataset_name, test_dataset_name), "test.json")}
        ).shuffle(seed=seed)        
        
        if "nli" in data_args.data_path:
            test_dataset = base_test_dataset.map(preprocess_function_nli, batched=True)
            test_dataset = test_dataset.remove_columns(column_names=["premise", "hypothesis"])
        
        elif "imdb" in data_args.data_path:
            test_dataset = base_test_dataset.map(preprocess_function, batched=True)
            test_dataset = test_dataset.remove_columns(column_names=["text"])
            
        else:
            raise NotImplementedError()
        
        test_datasets.append(test_dataset["test"])
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset["test"],
            shuffle=True,
            collate_fn=data_collator,
            batch_size=training_args.per_gpu_train_batch_size,
            drop_last=training_args.dataloader_drop_last,
            pin_memory=training_args.dataloader_pin_memory,
        )
        
        splitted_output_path = training_args.output_dir.split("/")
        splitted_output_path_last = splitted_output_path[-1].split("-")
        splitted_output_path_last[1] = test_dataset_name
        test_output_path = "/".join(splitted_output_path[:-1] + ["-".join(splitted_output_path_last)])
        
        os.makedirs(test_output_path, exist_ok=True)
        
        test_results = evaluation(
            steps=None,
            model=best_model,
            dataloader=test_dataloader,
            label_map=id2label,
            output_path=test_output_path,
            calibration_algorithm=data_args.calibration_algorithm,
            device=device,
            writer=None,
            num_monte_carlo=data_args.num_monte_carlo,
            split="test",
            seed=seed,
        )

        test_results.update({"learning_rate": lr_scheduler.get_last_lr()[0]})
        if re.search(r"TemperatureScaled", model_args.model_type):
            test_results.update({"temperature": temperature})
            
        elif re.search(r"(AutoTuneFaissKNN|AutoTuneLabelRejection)", model_args.model_type):
            test_results.update(
                {
                    "alpha": parameters.alpha,
                    "knn_temperature": parameters.temperature,
                    "lambda": parameters.lamb,
                    "bias": parameters.bias,
                }
            )
            
        elif re.search(r"AutoTuneDAC", model_args.model_type):
            for i, w in enumerate(parameters):
                test_results.update(
                    {
                        "weight" + str(i + 1): w
                    }
                )

        output_test_results_file = os.path.join(
            test_output_path, f"test_results_{str(seed)}.txt"
        )
        with open(output_test_results_file, "w") as f:
            logger.info("***** Test results *****")
            for key, value in test_results.items():
                logger.info("  %s = %s", key, value)
                f.write("%s = %s\n" % (key, value))
                
    concatenated: datasets.Dataset = datasets.concatenate_datasets(test_datasets)
    ood_labels = [0] * len(test_datasets[0]) + [1] * len(test_datasets[1])
    ood_test_dataset = concatenated.add_column("ood_label", ood_labels)
    
    data_collator_for_ood = DataCollatorWithPaddingForOOD(
        tokenizer=tokenizer,
        padding="longest",
        max_length=data_args.max_seq_length,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    
    test_ood_dataloader = torch.utils.data.DataLoader(
        ood_test_dataset,
        shuffle=True,
        collate_fn=data_collator_for_ood,
        batch_size=training_args.per_gpu_train_batch_size,
        drop_last=training_args.dataloader_drop_last,
        pin_memory=training_args.dataloader_pin_memory,
    )
    
    ood_test_results = evaluation(
        steps=None,
        model=best_model,
        dataloader=test_ood_dataloader,
        label_map=id2label,
        output_path=test_output_path,
        calibration_algorithm=data_args.calibration_algorithm,
        device=device,
        writer=None,
        num_monte_carlo=data_args.num_monte_carlo,
        split="test",
        seed=seed,
        is_ood_detection=True
    )
    
    output_ood_detection_test_results_file = os.path.join(
        training_args.output_dir, f"test_results_ood_detection_{str(seed)}.txt"
    )
    
    with open(output_ood_detection_test_results_file, "w") as f:
        logger.info("***** OOD Detection Test results *****")
        for key, value in ood_test_results.items():
            logger.info("  %s = %s", key, value)
            f.write("%s = %s\n" % (key, value))


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
        for i in range(1, 6):  # 1 ~ 10
            main(model_args, data_args, training_args, seed=i)
    else:
        main(model_args, data_args, training_args)
