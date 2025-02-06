from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from transformers.utils import ModelOutput


@dataclass
class SequenceClassifierOutputConf(ModelOutput):
    """
    Base class for outputs of sequence classification models.
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
    sngp_covariances: Optional[torch.Tensor] = None
    distances: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None
    distance_term: Optional[np.ndarray] = None
    label_term: Optional[np.ndarray] = None
    neighbor_scores: Optional[np.ndarray] = None
    uncertainty_scores: Optional[np.ndarray] = None


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
        ood_labels: Optional[np.ndarray] = None,
        uncertainty_scores: Optional[np.ndarray] = None
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs
        self.losses = losses
        self.tags = tags
        self.attentions = attentions
        self.ood_labels = ood_labels
        self.uncertainty_scores = uncertainty_scores

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
