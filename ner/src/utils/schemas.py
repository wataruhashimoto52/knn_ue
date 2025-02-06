from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np
import torch
from transformers.utils import ModelOutput


@dataclass
class TokenClassifierOutputConf(ModelOutput):
    """
    Base class for outputs of token classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed
            or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed
            or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        confidences (`torch.FloatTensor`): confidences of each inference
        losses: (`torch.FloatTensor`): loss of each instances (optional)
        tags: (`list[list[int]]`): best sequence (only for CRF)
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    confidences: Optional[torch.FloatTensor] = None
    full_confidences: Optional[torch.FloatTensor] = None
    losses: Optional[torch.FloatTensor] = None
    tags: Optional[list[list[int]]] = None
    span_logits: Optional[torch.Tensor] = None
    span_confidences: Optional[torch.Tensor] = None
    sequence_output: Optional[torch.Tensor] = None
    datastore_records: Optional[list[dict]] = None
    distances: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None


class EvalPredictionV2:
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
        losses: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        tags: Optional[list[list[int]]] = None,
        attentions: Optional[torch.Tensor] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.tags = tags
        self.attentions = attentions

    def __iter__(self):
        if (
            self.inputs is not None
            and self.losses is not None
            and self.tags is not None
        ):
            return iter(
                (self.predictions, self.label_ids, self.inputs, self.losses, self.tags)
            )
        elif self.inputs is not None and self.losses is not None:
            return iter((self.predictions, self.label_ids, self.inputs, self.losses))
        elif self.inputs is not None and self.tags is not None:
            return iter((self.predictions, self.label_ids, self.inputs, self.tags))
        elif self.losses is not None and self.tags is not None:
            return iter((self.predictions, self.label_ids, self.losses, self.tags))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx: int):
        if idx < 0 or idx > 4:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 3 and self.losses is None:
            raise IndexError("index out of range")
        if idx == 4 and self.tags is None:
            raise IndexError("index out of range")

        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        elif idx == 3:
            return self.losses
        elif idx == 4:
            return self.tags


@dataclass
class UncertaintyArguments:
    unc_method: str = field(
        default=None, metadata={"help": "Flair NER dataset name. [slpn,dp]"}
    )
    input_dims: str = field(
        default="[768]",
        metadata={"help": "the input dim to the latent embedding encoder"},
    )
    hidden_dims: str = field(
        default="[100]",
        metadata={"help": "the hidden dim to the latent embedding encoder"},
    )
    latent_dim: int = field(
        default=50,
        metadata={"help": "the output (latent) dim to the latent embedding encoder"},
    )
    output_dim: int = field(
        default=17,
        metadata={"help": "the output (latent) dim to the latent embedding encoder"},
    )
    k_lipschitz: float = field(
        default=None,
        metadata={"help": "Lipschitz constant. float or None (if no lipschitz)"},
    )
    kernel_dim: str = field(
        default=None, metadata={"help": "kernal size for comnv archi"}
    )
    no_density: bool = field(
        default=None, metadata={"help": "Use density estimation or not. boolean"}
    )
    density_type: str = field(
        default="radial_flow", metadata={"help": "Density type. string"}
    )
    n_density: int = field(
        default=8, metadata={"help": "# Number of density components. int"}
    )
    budget_function: str = field(
        default="id",
        metadata={"help": "Budget function name applied on class count. name"},
    )
    unc_seed: int = field(
        default=123,
        metadata={"help": "# seed of random among uncertainty quantification"},
    )
    radial_layers: int = field(
        default=10, metadata={"help": "# number of radial_layers in normalize flow"}
    )
    maf_layers: int = field(
        default=0, metadata={"help": "# number of maf_layers in normalize flow"}
    )
    gaussian_layers: int = field(
        default=0, metadata={"help": "# number of gaussian_layers in normalize flow"}
    )
    use_batched_flow: bool = field(default=True, metadata={"help": ""})
    alpha_evidence_scale: str = field(
        default="latent-new",
        metadata={
            "help": " ['latent-old', 'latent-new', 'latent-new-plus-classes', None]"
        },
    )
    prior_mode: str = field(
        default="global", metadata={"help": " ['global', 'local', 'global_local']"}
    )
    neighbor_mode: str = field(
        default=None,
        metadata={"help": " ['self_att', 'None', 'closest', 'simple_project']"},
    )
    use_uce: int = field(
        default=None,
        metadata={"help": "force using ce loss if 0, and force using uce loss if 1"},
    )
    only_test: int = field(
        default=0, metadata={"help": "1: only do test; 0: do both test and training"}
    )
    normalize_dis: int = field(
        default=1,
        metadata={
            "help": "1: do global/local distribution normalization; 0: donot do distri normalization"
        },
    )
    draw_tSNE: int = field(
        default=0, metadata={"help": "1: draw tSNE figure; 0: donot draw tSNE figure"}
    )
    self_att_dk_ratio: int = field(
        default=8, metadata={"help": "dk = class_num / self_att_dk_ratio"}
    )
    self_att_droput: float = field(
        default=0.05, metadata={"help": "dropout ratio in self_att"}
    )
    cal_unique_predict_scores: bool = field(
        default=False,
        metadata={
            "help": "whether calculate and use the unique predicted scores in the testing process"
        },
    )
    use_stable: bool = field(
        default=False,
        metadata={"help": "whether use stable skills to stable the training"},
    )
    use_var_metric: bool = field(
        default=True,
        metadata={"help": "whether use stable skills to stable the training"},
    )

    # below is related to ood detection task
    te_task: str = field(default="mis", metadata={"help": " ['mis', 'ood']"})
    oodset_name: str = field(
        default="", metadata={"help": "Dataset in OOD usage for Flair NER dataset."}
    )
    ood_ratio: float = field(
        default=0.5,
        metadata={"help": "the ratio of OOD samples among the original samples"},
    )
    leave_out_labels: str = field(
        default="['Price', 'Hours']", metadata={"help": "leave out labels"}
    )
    exclude_split_ratio: str = field(
        default="[0.8, 0.9, 1.0]",
        metadata={"help": "the ratio used to split excluded split ratio"},
    )
    ood_eval_mode: str = field(
        default="entity_ori",
        metadata={
            "help": "entity_ori: use the default entity_eval; entity_fp: take the ws as fp; token_ori: use the default token_eval"
        },
    )
    ### when ood_eval_mode is in [entity_fp, token_ori], the wrong_span_based scores are uselessful, please ignore them.

    # below is related to sequential training
    use_seq_training: bool = field(
        default=False, metadata={"help": "whether use and the sequential loss"}
    )
    pretr_ep_num: int = field(
        default=20,
        metadata={"help": "init epoch used for the training pre_trained model only"},
    )
    load_pretr_bert_latentenc: bool = field(
        default=False, metadata={"help": "whether load the pretr_bert_latentenc or not"}
    )
    well_fined_model_path: str = field(
        default=None, metadata={"help": "the path to load pretr_bert_latentecn_emb"}
    )
    pretr_lr: float = field(
        default=0.001, metadata={"help": "the lr rate in the pre-training"}
    )
    pretr_weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for optimizer in pretraining."}
    )

    # below is related to combine training
    use_multitask_training: bool = field(
        default=False, metadata={"help": "whether use and the multi_task loss"}
    )
    bert_loss_w: float = field(
        default=0.1, metadata={"help": "weight of using the bert loss"}
    )

    # below is baseline - dropout related parameters
    main_dropout: float = field(
        default=0.00, metadata={"help": "dropout ratio in framework"}
    )  # 0.15
    test_dropout_num: int = field(
        default=10, metadata={"help": "dropout time in the testing process"}
    )
