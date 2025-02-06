
import time
import copy
import math
from enum import Enum, auto
from typing import Final, Optional, Tuple, Union


import faiss
from collections import defaultdict
import numpy as np
import torch
from scipy import optimize
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataclasses import dataclass
from einops import rearrange
from torch.nn import CrossEntropyLoss
from transformers import (
    DebertaV2ForSequenceClassification,
    DebertaV2Model,
    PretrainedConfig,
    activations
)

from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, StableDropout

from algorithms.density_estimators import DensityEstimatorWrapper, NormalizingFlow
from algorithms.evidential_nn import BeliefMatchingLoss
from algorithms.label_smoothing import LabelSmoother
from algorithms.spectral_modules import spectral_norm_fc
from algorithms.mahalanobis_distance import compute_centroids, compute_covariance, mahalanobis_distance
from algorithms.spectral_normalization import spectral_norm
from algorithms.nuq.nuq_classifier import NuqClassifier
from utils.schemas import SequenceClassifierOutputConf
from utils.data_utils import prepare_input


class IndexType(Enum):
    FLAT = auto()
    IVF = auto()
    PQ = auto()
    IVFPQ = auto()
    HNSW = auto()
    HNSW_PQ = auto()
    HNSW_IVFPQ = auto()

    
@dataclass
class kNNUEIntermediateOutput:
    logits: np.ndarray
    distances: np.ndarray
    indices: np.ndarray
    distance_term: np.ndarray
    label_term: Optional[np.ndarray]
    neighbor_scores: Optional[np.ndarray]
    

@dataclass
class kNNParameters:
    alpha: float
    temperature: float
    lamb: float
    bias: float
    
    
def np_softmax(x: np.ndarray) -> np.ndarray:
    max = np.max(
        x, axis=1, keepdims=True
    )  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(
        e_x, axis=1, keepdims=True
    )  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x


def apply_dropout(m: nn.Module):
    if type(m) == nn.Dropout:
        m.train()
        
        
__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}


class BaselineDeBERTaV3ForSequenceClassification(DebertaV2ForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)


        loss = None
        losses = None
        confidences = None
        if labels is not None:

            confidences = F.softmax(logits, dim=-1)
            
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            sequence_output=pooled_output,
        )


class TemperatureScaledDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        tau: float = 1.0
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.tau = tau
        
        self.tol = 1e-12
        self.eps = 1e-7
        self.disp = False
        
        # scaling parameter, temperature parameter.
        self.bnds = [[0, 5.0]]
        self.init = [tau]
        
        self.is_test = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test
        
    def get_optimized_temperature(self, val_loader: torch.utils.data.DataLoader) -> float:
        logits_list = []
        labels_list = []
        device = self.deberta.device
        with torch.no_grad():
            for batch in val_loader:
                batch = prepare_input(batch, device)
                logits = self.forward_logits(**batch)
                logits_list.append(logits)
                labels_list.append(batch["labels"])
            
        logits = torch.cat(logits_list).detach().cpu().numpy()
        labels = torch.cat(labels_list).detach().cpu().numpy()
        
        st = time.time()
        params = optimize.minimize(
            self.ll_lf,
            self.init,
            args=(logits, labels),
            method="L-BFGS-B",
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp},
        )
        ed = time.time()
        
        print("Temperature Scaling's temperature optimization done!: ({} sec)".format(ed - st))

        self.tau: float = params.x[0]
        return float(self.tau)
    
    def forward_logits(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)
        return logits
    
    def temperature_scale(self, logits: np.ndarray, temperature: list[float]) -> torch.Tensor:
        return logits / temperature[0]
    
    def ll_lf(
        self,
        temperature: list[float],
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        logits = self.temperature_scale(logits, temperature)
        probs = np_softmax(logits)
        onehot_labels = np.eye(probs.shape[-1])[labels]
        N = probs.shape[0]
        ce: np.ndarray = -np.sum(onehot_labels * np.log(probs + 1e-12)) / N
        return ce
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        
        logits: torch.Tensor = self.classifier(pooled_output)
        if self.is_test:
            logits = torch.from_numpy(
                self.temperature_scale(logits.detach().cpu().numpy(), [self.tau])
            ).to(self.deberta.device)
        

        loss = None
        losses = None
        confidences = None
        if labels is not None:

            confidences = F.softmax(logits, dim=-1)
            
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            sequence_output=pooled_output,
        )


class LabelSmoothingDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(self, config: PretrainedConfig, smoothing: float) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        self.smoothing = smoothing
        self.criterion_none = LabelSmoother(reduction="none", epsilon=self.smoothing)
        self.criterion_mean = LabelSmoother(reduction="mean", epsilon=self.smoothing)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)


        loss = None
        losses = None
        confidences = None
        if labels is not None:

            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = self.criterion_none(
                logits, labels
            )
            loss: torch.Tensor = self.criterion_mean(
                logits, labels
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
        )


class MCDropoutDeBERTaV3ForSequenceClassification(DebertaV2ForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(reduction="none")
            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class SpectralNormalizedDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self, config: PretrainedConfig, spectral_norm_upper_bound: float
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        """
        for layer in self.deberta.encoder.layer:
            layer.intermediate.dense = spectral_norm(layer.intermediate.dense)
            layer.output.dense = spectral_norm(layer.output.dense)
            
        """
        for layer in self.deberta.encoder.layer:
            layer.intermediate.dense = spectral_norm_fc(
                layer.intermediate.dense,
                spectral_norm_upper_bound,
            )
            layer.output.dense = spectral_norm_fc(
                layer.output.dense,
                spectral_norm_upper_bound,
            )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:

            loss_fct = CrossEntropyLoss(reduction="none")
            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = loss_fct(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(
                logits.view(-1, self.num_labels), labels.view(-1)
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class BeliefMatchingDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self, config: PretrainedConfig, coeff: float = 1e-2, prior: float = 1.0
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.coeff = coeff
        self.prior = prior

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:

            loss_fct = BeliefMatchingLoss(coeff=self.coeff, prior=self.prior)
            confidences = F.softmax(logits, dim=-1)
            inputs = logits.view(-1, self.num_labels)
            targets = labels.view(-1)
            inputs = torch.stack(
                [i for i, t in zip(inputs, targets) if t != -100], dim=0
            )
            targets = torch.stack([t for t in targets if t != -100], dim=0)
            inputs.to(device=logits.device)
            targets.to(device=labels.device)
            losses: torch.Tensor = loss_fct(inputs, targets)
            loss: torch.Tensor = losses.mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class DensitySoftmaxDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        density_estimator: DensityEstimatorWrapper,
        max_log_prob: float,
        feature_normalization: bool,
        device: torch.device,
        pca_model: Optional[faiss.VectorTransform] = None,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.density_estimator = density_estimator
        self.max_log_prob = max_log_prob
        self.feature_normalization = feature_normalization
        self.device_type = device
        self.pca_model = pca_model

        self.normalizer = lambda x: x / (
            torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            with torch.inference_mode():
                if self.feature_normalization:
                    log_probs = self.density_estimator.log_prob(
                        self.normalizer(pooled_output)
                    )
                else:
                    if self.pca_model:
                        log_probs = self.density_estimator.log_prob(
                            torch.from_numpy(
                                self.pca_model.apply(
                                    pooled_output.detach().cpu().numpy()
                                )
                            ).to(logits.device)
                        )
                    else:
                        log_probs = self.density_estimator.log_prob(pooled_output)
            normalized_probs = torch.exp(log_probs / self.max_log_prob)

            dst_logits: torch.Tensor = logits * torch.sigmoid(
                normalized_probs.unsqueeze(1)
            )

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(dst_logits, labels)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(dst_logits, labels)
            
            confidences = F.softmax(dst_logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )
        
        
class PosteriorNetworksDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        label2count: list[int],
        feature_normalization: bool,
        device: torch.device,
        budget_function: str = "id",
        density_type: str = "radial_flow",
        latent_dim: int = 10,
        n_density: int = 8,
        pca_model: Optional[faiss.VectorTransform] = None,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, latent_dim)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        self.density_type = density_type
        self.label2count = label2count
        self.feature_normalization = feature_normalization
        self.device_type = device
        self.pca_model = pca_model

        self.normalizer = lambda x: x / (
            torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )
        self.regr = 1e-5
        
        if self.density_type in ("planar_flow", "radial_flow"):
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlow(dim=latent_dim, flow_length=n_density, flow_type=density_type)
                    for c in range(self.config.num_labels)
                ]
            )
        else:
            raise NotImplementedError

        # Initialize weights and apply final processing
        
        if budget_function in __budget_functions__:
            self.N, self.budget_function = __budget_functions__[budget_function](label2count), budget_function
        else:
            raise NotImplementedError
        
        self.post_init()
        
    def compute_uce_loss(self, alpha: torch.Tensor, soft_output: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
        with torch.autograd.detect_anomaly():
            alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.config.num_labels)
            entropy_reg = torch.distributions.Dirichlet(alpha).entropy()
            if reduce == "sum":
                UCE_loss = torch.sum(soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * torch.sum(entropy_reg)
            else:
                UCE_loss = (soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))) - self.regr * entropy_reg

            return UCE_loss

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        pooled_output: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        batch_size = pooled_output.shape[0]
        # if self.N.device != self.deberta.device:
        #     self.N = self.N.to(self.deberta.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N
        
        if labels is not None:
            if self.feature_normalization:
                query_representation: torch.Tensor = self.normalizer(pooled_output)
            else:
                query_representation: torch.Tensor = pooled_output
            
            log_q_zk = torch.zeros((batch_size, self.config.num_labels)).to(self.deberta.device)
            alpha = torch.zeros((batch_size, self.config.num_labels)).to(self.deberta.device)
            for c in range(self.config.num_labels):
                log_probs = self.density_estimation[c].log_prob(query_representation)
                log_q_zk[:, c] = log_probs
                alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
                
            dst_logits = torch.nn.functional.normalize(alpha, p=1)
            
            onehot_labels = F.one_hot(labels, self.config.num_labels)

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(dst_logits, labels)
            loss: torch.Tensor = self.compute_uce_loss(dst_logits, onehot_labels)
            
            confidences = F.softmax(dst_logits, dim=-1)


        if not return_dict:
            output = (dst_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=dst_logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class AutoTuneFaissKNNDeBERTaV3ForSequenceClassification(BaselineDeBERTaV3ForSequenceClassification):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        feature_normalization: bool,
        encoded_train_samples: np.ndarray,
        topk: int,
        use_pca: bool,
        use_recomp: bool,
        index_type: str,
        alpha: float = 1.0,
        knn_temperature: int = 1000,
        transform_dim: int = 256,
        nprobe: int = 16,
        n_list: int = 100,
        n_subvectors: int = 64,
        use_neighbor_labels: bool = False,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        
        self.feature_normalization = feature_normalization
        self.topk = topk
        self.alpha = alpha * torch.ones(1).to(device)
        self.knn_temperature = knn_temperature * torch.ones(1).to(device)
        self.use_pca = use_pca
        self.use_recomp = use_recomp
        self.index_type = index_type
        self.nprobe = nprobe
        self.n_list = n_list
        self.n_subvectors = n_subvectors
        self.transform_dim = transform_dim
        self.encoded_train_samples = encoded_train_samples
        self.device_type = device
        self.is_test = False
        self.use_neighbor_labels = use_neighbor_labels
        
        self.tol = 1e-12
        self.eps = 1e-7
        self.disp = False
        self.opt_maxiter = 1000
        
        # scaling parameter, temperature parameter.
        if self.use_neighbor_labels:
            self.bnds = [[0, 3.0]] + [[0, 20.0]] + [[0, 3.0]] + [[-3.0, 3.0]]
            self.init = [1.0] + [1.0] + [1.0] + [1.0]
            
            """self.bnds = [[0, 3.0]] + [[0, 5.0]] + [[-3.0, 3.0]]
            self.init = [1.0] + [1.0] + [1.0]"""
        else:
            self.bnds = [[0, 3.0]] + [[0, 20.0]]
            self.init = [1.0] + [1.0]
        
        self.weights: list[float] = []

        # torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        self.normalizer = lambda x: x / (
            np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def build_index(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        embeddings = []
        cached_labels = []
        batches = tqdm(train_dataloader)
        device = self.deberta.device
        for batch in batches:
            
            batch = prepare_input(batch, device)
            with torch.inference_mode():
                outputs: SequenceClassifierOutputConf = self.forward(**batch)
                
            cached_labels.append(batch["labels"].detach().cpu().numpy())
            embeddings.append(outputs.sequence_output.detach().cpu().numpy())
        
        encoded_train_samples: np.ndarray = np.concatenate(embeddings)
        train_cached_labels: np.ndarray = np.concatenate(cached_labels)
        
        if self.feature_normalization:
            encoded_train_samples = self.normalizer(encoded_train_samples)
        
        dim = self.transform_dim if self.use_pca else self.config.hidden_size
        
        
        if self.index_type == IndexType.FLAT.name:
            base_index = faiss.IndexFlatL2(dim)
        elif self.index_type == IndexType.HNSW.name:
            base_index = faiss.IndexHNSWFlat(dim, 32)
            base_index.hnsw.efConstruction = 128
            
        elif self.index_type == IndexType.PQ.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFPQ(quantizer, dim, 1, self.n_subvectors, 8)
            
        elif self.index_type == IndexType.IVF.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFFlat(quantizer, dim, self.n_list, faiss.METRIC_L2)
            
        elif self.index_type == IndexType.IVFPQ.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFPQ(quantizer, dim, self.n_list, self.n_subvectors, 8)
            
            
        elif self.index_type == IndexType.HNSW_PQ.name:
            base_index = faiss.IndexHNSWPQ(dim, self.M, 32)
        elif self.index_type == IndexType.HNSW_IVFPQ.name:
            quantizer = faiss.IndexHNSWFlat(self.transform_dim, 32)
            base_index = faiss.IndexIVFPQ(quantizer, self.transform_dim, self.config.num_labels, self.n_subvectors, 8)
        else:
            raise NotImplementedError()
        
        # train pca.
        if self.use_pca:
            self.vtrans = faiss.PCAMatrix(self.config.hidden_size, self.transform_dim)
            self.vtrans.train(encoded_train_samples)
        else:
            self.vtrans = None

        if self.use_pca:
            base_index = faiss.IndexPreTransform(self.vtrans, base_index)
        
        if "IVFPQ" in self.index_type:
            base_index.nprobe = self.nprobe
        
        if self.device_type.type == "cuda":
            if "PQ" in self.index_type:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, base_index, co)
            else:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, base_index)
                
            base_index.reset()

        elif self.device_type.type == "cpu":
            self.index = base_index
        else:
            raise NotImplementedError()
        
        if not self.index.is_trained:
            self.index.train(encoded_train_samples)
            
        self.index.add(encoded_train_samples)
        self.train_cached_labels = train_cached_labels

        
    def forward_logits(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)
        
        return logits, pooled_output
    
    def knn_weighting(self, logits: torch.Tensor, pooled_output: torch.Tensor) -> torch.Tensor:
        if self.feature_normalization:
            D, I = self.index.search(
                self.normalizer(pooled_output).detach().cpu().numpy(),
                self.topk,
            )
            
        else:
            D, I = self.index.search(
                pooled_output.detach().cpu().numpy(), self.topk
            )

        if self.use_recomp:
            knn_weight = torch.exp(
                -(
                    torch.cdist(
                        torch.from_numpy(self.encoded_train_samples[I]).to(
                            logits.device
                        ),
                        pooled_output.unsqueeze(1),
                        p=2,
                    ).squeeze()
                    / self.knn_temperature
                ).mean(-1)
            )
        else:
            distances = torch.from_numpy(D).to(logits.device).divide(self.knn_temperature)
            knn_weight = self.alpha * torch.exp(-distances).mean(-1).to(logits.device)

        dst_logits: torch.Tensor = logits * knn_weight.unsqueeze(1)
        
        return dst_logits
    
    def knn_weighting_numpy(
            self,
            logits: np.ndarray,
            pooled_output: np.ndarray,
            params: list[float],
        ) -> kNNUEIntermediateOutput:
        if self.feature_normalization:
            query_vector = self.normalizer(pooled_output)
        else:
            query_vector = pooled_output
            
        D, I = self.index.search(query_vector, self.topk)                

        if self.use_recomp:
            reshaped_indices = rearrange(I, "b k -> (b k)")
            extracted_raw_vectors = rearrange(self.encoded_train_samples[reshaped_indices], "(k b) d -> b k d",
                                              b=query_vector.shape[0], d=query_vector.shape[-1])
            
            dist = np.linalg.norm(
                np.expand_dims(query_vector, 0) - \
                    rearrange(extracted_raw_vectors, "b k d -> k b d", b=query_vector.shape[0], k=self.topk),
                axis=-1
            )
            distances = rearrange("k b -> b k", dist) / params[1]
        else:  # use quantized distance
            distances: np.ndarray = D / params[1]  # [b, k]
            
        distance_term = params[0] * np.exp(-distances).mean(-1)
        
        knn_weight = distance_term
        
        label_term = None
        neighbor_scores = None
        
        if self.use_neighbor_labels:
            pred_labels = logits.argmax(-1)  # [B, ]
            values = rearrange(I, "b k -> (b k)")

            neighbor_labels = rearrange(self.train_cached_labels[values], "(b k) -> b k", b=logits.shape[0], k=self.topk)  # [B, K]
            neighbor_scores = np.sum(neighbor_labels == pred_labels.reshape(-1, 1), axis=1)
            # neighbor_scores = np.array(
            #     [(pred_labels[bt] == neighbor_labels[bt]).sum() for bt in range(logits.shape[0])]
            # )
            # neighbor_scores = np.sum(pred_labels == neighbor_labels, axis=1)
            
            label_term = params[2] * (neighbor_scores / self.topk + params[3])
            
            knn_weight += label_term
            
            
        dst_logits = logits * np.expand_dims(knn_weight, 1)
        
        return kNNUEIntermediateOutput(
            logits=dst_logits,
            distances=D,
            indices=I,
            distance_term=distance_term,
            label_term=label_term,
            neighbor_scores=neighbor_scores,
        )
    
    def ll_lf(
        self,
        weights: np.ndarray,
        logits: np.ndarray,
        hidden_states_list: np.ndarray, 
        labels: np.ndarray, 
    ) -> np.ndarray:
        knn_internal_output = self.knn_weighting_numpy(logits, hidden_states_list, weights)
        probs = np_softmax(knn_internal_output.logits)
        onehot_labels = np.eye(probs.shape[-1])[labels]
        N = probs.shape[0]
        ce: np.ndarray = -np.sum(onehot_labels * np.log(probs + 1e-12)) / N
        return ce
    
    def optimize_knn_parameters_numpy(self, val_loader: torch.utils.data.DataLoader) -> kNNParameters:
        logits_list = []
        pooled_output_list = []
        labels_list = []
        device = self.deberta.device
        
        # cache logits and labels.
        with torch.no_grad():
            for batch in val_loader:
                batch = prepare_input(batch, device)
                # reshaped_logits: [B * L, K]
                # reshaped_sequence_output: [B * L, D]
                logits, sequence_output = self.forward_logits(**batch)
                logits_list.append(logits)
                pooled_output_list.append(sequence_output)
                labels_list.append(batch["labels"])
            
        logits = torch.cat(logits_list).detach().cpu().numpy()
        pooled_outputs = torch.cat(pooled_output_list).detach().cpu().numpy()
        labels = torch.cat(labels_list).detach().cpu().numpy()
        
        st = time.time()
        params = optimize.minimize(
            self.ll_lf,
            self.init,
            args=(logits, pooled_outputs, labels),
            method="L-BFGS-B",
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp},
        )
        ed = time.time()

        print("kNN UE Optimization done!: ({} sec)".format(ed - st))
        
        self.weights = params.x
        
        if self.use_neighbor_labels:
            return kNNParameters(
                alpha=float(self.weights[0]),
                temperature=float(self.weights[1]),
                lamb=float(self.weights[2]),
                bias=float(self.weights[3]),
            )
        return kNNParameters(
                alpha=float(self.weights[0]),
                temperature=float(self.weights[1]),
                lamb=1.0,
                bias=1.0,
            )
        
    def set_knn_temperature(self, temperature: Union[int, float, torch.Tensor]) -> None:
        self.knn_temperature = float(temperature) * torch.ones(1).to(self.deberta.device)
        
    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test
    
    def optimize_knn_parameters(self, val_loader: torch.utils.data.DataLoader) -> kNNParameters:
        logits_list = []
        pooled_output_list = []
        labels_list = []
        device = self.deberta.device
        
        # cache logits and labels.
        with torch.no_grad():
            for batch in val_loader:
                batch = prepare_input(batch, device)
                logits, pooled_output = self.forward_logits(**batch)
                logits_list.append(logits)
                pooled_output_list.append(pooled_output)
                labels_list.append(batch["labels"])
            
            logits = torch.cat(logits_list).to(device)
            pooled_outputs = torch.cat(pooled_output_list).to(device)
            labels = torch.cat(labels_list).to(device)
        
        nll_criterion = nn.CrossEntropyLoss()

        # optimize knn temperature.
        temperatures = list(range(500, 2600, 100))
        temp2loss: dict[int, float] = {}
        for temp in temperatures:
            self.set_knn_temperature(temp)
            val_loss: torch.Tensor = nll_criterion(self.knn_weighting(logits, pooled_outputs), labels)
            temp2loss[temp] = float(val_loss.item())
            
        print(temp2loss)
        final_temperature: float = min(temp2loss, key=temp2loss.get)
        self.set_knn_temperature(final_temperature)
            
        # next, optimize alpha.
        self.alpha = nn.Parameter(self.alpha)
        optimizer = torch.optim.LBFGS([self.alpha], lr=0.01, max_iter=100) 
        
        def eval():
            optimizer.zero_grad()
            loss: torch.Tensor = nll_criterion(self.knn_weighting(logits, pooled_outputs), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return kNNParameters(
            alpha=float(self.alpha),
            temperature=float(self.knn_temperature),
        )
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        distances = None
        indices = None
        distance_term = None
        label_term = None
        neighbor_scores = None
        
        uncertainty_scores = None
        
        if labels is not None:
            if self.is_test:
                knn_internal_output = self.knn_weighting_numpy(
                    logits=logits.detach().cpu().numpy(),
                    pooled_output=pooled_output.detach().cpu().numpy(),
                    params=self.weights,
                )
                logits: torch.Tensor = torch.from_numpy(knn_internal_output.logits).to(self.deberta.device)
                distances = knn_internal_output.distances
                indices = knn_internal_output.indices
                distance_term = knn_internal_output.distance_term
                label_term = knn_internal_output.label_term
                neighbor_scores = knn_internal_output.neighbor_scores

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)

            confidences = F.softmax(logits, dim=-1)
            
            # uncertainty_scores: np.ndarray = torch.distributions.Categorical(confidences).entropy().detach().cpu().numpy()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=pooled_output,
            distances=distances,
            indices=indices,
            distance_term=distance_term,
            label_term=label_term,
            neighbor_scores=neighbor_scores,
            uncertainty_scores=uncertainty_scores,
        )


def BertLinear(i_dim, o_dim, bias=True):
    m = nn.Linear(i_dim, o_dim, bias)
    nn.init.normal_(m.weight, std=0.02)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def RandomFeatureLinear(i_dim, o_dim, bias=True, require_grad=False):
    m = nn.Linear(i_dim, o_dim, bias)
    # https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/bert_sngp.py
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    # freeze weight
    m.weight.requires_grad = require_grad
    if bias:
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # freeze bias
        m.bias.requires_grad = require_grad
    return m


class SpectralNormalizedContextPooler(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        n_power_iterations: int,
        norm_bound: float,
    ):
        super().__init__()
        self.dense = spectral_norm(
            BertLinear(config.pooler_hidden_size, config.pooler_hidden_size),
            n_power_iterations=n_power_iterations,
            norm_bound=norm_bound,
        )
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = activations.ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


class SNGPDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        gp_kernel_scale: float = 1.0,
        num_inducing: int = 1024,
        gp_output_bias: float = 0.,
        layer_norm_eps: float = 1e-12,
        n_power_iterations: int = 1,
        spec_norm_bound: float = 0.95,
        scale_random_features: bool = True,
        normalize_input: bool = True,
        gp_cov_momentum: float = 0.999,
        gp_cov_ridge_penalty: float = 1e-3,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = SpectralNormalizedContextPooler(
            config,
            n_power_iterations,
            spec_norm_bound,
        )
        output_dim = self.pooler.output_dim

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        self.is_test = False
        
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum

        self.pooled_output_dim = self.pooler.output_dim

        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input

        self._gp_input_normalize_layer = torch.nn.LayerNorm(self.pooler.output_dim, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_labels, bias=False)
        # bert gp_output_bias_trainable is false
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L69
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * num_labels).to(device)
        self._random_feature = RandomFeatureLinear(self.pooled_output_dim, num_inducing)

        # Laplace Random Feature Covariance
        # Posterior precision matrix for the GP's random feature coefficients.
        self.initial_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing).to(device))
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)
        
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def set_is_test(self, is_test: bool) -> bool:
        self.is_test = is_test
        
    def compute_predictive_covariance(self, gp_feature: torch.Tensor) -> torch.Tensor:
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        # Computes the covariance matrix of the feature coefficient.
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix
        
    def gp_layer(self, gp_inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # gp_inputs: [batch_size, embedding_dim]
        # Supports lengthscale for custom random feature layer by directly
        # rescaling the input.
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        gp_feature = self._random_feature(gp_inputs)
        # cosine
        gp_feature = torch.cos(gp_feature)

        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale
            
        # print("gp_feature: ", gp_feature.shape) [batch_size, gp_hidden_dim (batch_size x max_len)]

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias
        
        # print("gp_output: ", gp_output.shape) [batch_size, ?]

        if self.training:
            # update precision matrix
            self.update_cov(gp_feature)
        return gp_feature, gp_output
    
    @staticmethod
    def mean_field_logits(
        logits: torch.Tensor,
        covmat: Optional[torch.Tensor] = None,
        mean_field_factor: float = 1.,
        likelihood: str = 'logistic',
    ):
        """Adjust the model logits so its softmax approximates the posterior mean [1].
        Arguments:
        logits: A float tensor of shape (batch_size, num_classes).
        covmat: A float tensor of shape (batch_size, batch_size). If None then it
        assumes the covmat is an identity matrix.
        mean_field_factor: The scale factor for mean-field approximation, used to
        adjust the influence of posterior variance in posterior mean
        approximation. If covmat=None then it is used as the scaling parameter for
        temperature scaling.
        likelihood: Likelihood for integration in Gaussian-approximated latent
        posterior.
        Returns:
        True or False if `pred` has a constant boolean value, None otherwise.
        """
        if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
            raise ValueError(
                f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
            )

        if mean_field_factor < 0:
            return logits

        # Compute standard deviation.
        if covmat is None:
            variances = 1.
        else:
            variances = torch.diagonal(covmat)

        # Compute scaling coefficient for mean-field approximation.
        if likelihood == 'poisson':
            logits_scale = torch.exp(- variances * mean_field_factor / 2.)
        else:
            logits_scale = torch.sqrt(1. + variances * mean_field_factor)

        # Cast logits_scale to compatible dimension.
        if len(logits.shape) > 1:
            logits_scale = torch.unsqueeze(logits_scale, dim=-1)

        return logits / logits_scale

    
    def update_cov(self, gp_feature: torch.Tensor) -> None:
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        # Updates the population-wise precision matrix.
        if self.gp_cov_momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision
            # matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.gp_cov_momentum * self.precision_matrix +
                    (1. - self.gp_cov_momentum) * precision_matrix_minibatch)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        #self.precision_matrix.weight = precision_matrix_new
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)
        
    def reset_cov(self) -> None:
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.initial_precision_matrix), requires_grad=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        
        gp_feature, logits = self.gp_layer(pooled_output)


        loss = None
        covariances = None
        losses = None
        confidences = None
        
        if self.is_test:
            covariances = self.compute_predictive_covariance(gp_feature)
            logits = self.mean_field_logits(
                logits,
                covariances,
                mean_field_factor=0.1,
                likelihood="logistic",
            )
            
        criterion_none = nn.CrossEntropyLoss(reduction="none")
        criterion_mean = nn.CrossEntropyLoss()
        
        if labels is not None:
            confidences = F.softmax(logits, dim=-1)
            losses: torch.Tensor = criterion_none(
                logits, labels
            )
            loss: torch.Tensor = criterion_mean(
                logits, labels
            )

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
        )


class AutoTuneDACDeBERTaV3ForSequenceClassification(BaselineDeBERTaV3ForSequenceClassification):
    M: Final[int] = 32  # 64, 32, 8
    NUM_HIDDEN_LAYERS: Final[int] = 13
    
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        feature_normalization: bool,
        topk: int,
        use_pca: bool,
        use_recomp: bool,
        use_only_last_layer: bool,
        index_type: str,
        alpha: float = 1.0,
        knn_temperature: int = 1000,
        transform_dim: int = 256,
        nprobe: int = 16,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        
        self.feature_normalization = feature_normalization
        self.use_only_last_layer = use_only_last_layer
        self.topk = topk
        self.alpha = alpha * torch.ones(1).to(device)
        self.knn_temperature = knn_temperature * torch.ones(1).to(device)
        self.use_pca = use_pca
        self.use_recomp = use_recomp
        self.index_type = index_type
        self.nprobe = nprobe
        self.transform_dim = transform_dim
        self.device_type = device
        self.is_test = False
        
        self.layer_wise_embeddings: dict[int, np.ndarray] = {}
        
        self.tol = 1e-7
        self.eps = 1e-7
        self.disp = False
        self.opt_maxiter = 600 # 1000
        
        if use_only_last_layer:
            num_hidden_layer = 1
        else:
            num_hidden_layer = self.NUM_HIDDEN_LAYERS
            
        self.temperature_weights = torch.rand(num_hidden_layer).to(device)
        self.temperature_bias = torch.rand(1).to(device)

        self.bnds = [[0, 10000.0]] * num_hidden_layer + [[-100.0, 100.0]]
        self.init = [1.0] * num_hidden_layer + [1.0]
        
        self.weights: list[float] = []

        self.normalizer = lambda x: x / (
            np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )

        # Initialize weights and apply final processing
        self.post_init()
        
    def build_layer_wise_embeddings(self, train_dataloader: torch.utils.data.DataLoader) -> None:
        embeddings = defaultdict(list)
        batches = tqdm(train_dataloader)
        device = self.deberta.device
        for batch in batches:
            
            batch = prepare_input(batch, device)
            with torch.inference_mode():
                outputs: SequenceClassifierOutputConf = self.forward(**batch)
                
            for i, hidden_state in enumerate(outputs.hidden_states):
                # hidden_states: [B, L, D]
                # hidden_state = hidden_state.mean(1)
                hidden_state = hidden_state[:, 0, :]
                embeddings[i].append(hidden_state.detach().cpu().numpy())
                
        layer_wise_embeddings = {
            i: np.concatenate(samples) for i, samples in embeddings.items()
        }
        if self.feature_normalization:
            layer_wise_embeddings = {
                i: self.normalizer(samples) for i, samples in layer_wise_embeddings.items()
            }
            
        self.layer_wise_embeddings = layer_wise_embeddings
        
    def build_index(self, samples: np.ndarray) -> faiss.Index:
        dim = self.transform_dim if self.use_pca else self.config.hidden_size
        
        if self.index_type == IndexType.FLAT.name:
            base_index = faiss.IndexFlatL2(dim)
        elif self.index_type == IndexType.HNSW.name:
            base_index = faiss.IndexHNSWFlat(dim, 32)
            base_index.hnsw.efConstruction = 128
            
        elif self.index_type == IndexType.PQ.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFPQ(quantizer, dim, 1, self.M, 8)
            
        elif self.index_type == IndexType.IVFPQ.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFPQ(quantizer, dim, 100, self.M, 8)
        elif self.index_type == IndexType.HNSW_PQ.name:
            base_index = faiss.IndexHNSWPQ(dim, self.M, 32)
        elif self.index_type == IndexType.HNSW_IVFPQ.name:
            quantizer = faiss.IndexHNSWFlat(self.transform_dim, 32)
            base_index = faiss.IndexIVFPQ(quantizer, self.transform_dim, self.config.num_labels, self.M, 8)
        else:
            raise NotImplementedError()
        
        # train pca.
        if self.use_pca:
            self.vtrans = faiss.PCAMatrix(self.config.hidden_size, self.transform_dim)
            self.vtrans.train(samples)
        else:
            self.vtrans = None
        
        if self.use_pca:
            base_index = faiss.IndexPreTransform(self.vtrans, base_index)
        
        if "IVF" in self.index_type:
            base_index.nprobe = self.nprobe
        
        index: faiss.Index = base_index
        
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
        if not index.is_trained:
            index.train(samples)
            
        index.add(samples)
        
        return index
        
    def forward_logits(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False if self.training else True,  # output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)
        
        return logits, outputs.hidden_states
    
    def knn_weighting_numpy(
        self,
        logits: np.ndarray,
        hidden_states: np.ndarray,
        weights: list[float],
    ) -> np.ndarray:
        """knn weighted temperature scaling

        Args:
            reshaped_logits (torch.Tensor): [B, K]
            reshaped_hidden_states (torch.Tensor) [B, N, H]
                B: batch size
                N: num of layers
                H: dim of hidden state

        Returns:
            torch.Tensor: _description_
        """
        batch_size = logits.shape[0]
        num_layers, dim_hidden_state = hidden_states.shape[1], hidden_states.shape[2]
        
        search_hidden_states = rearrange(hidden_states, "b n h -> n b h")
        if self.use_only_last_layer:
            num_layers = 1
            search_hidden_states = [search_hidden_states[-1]]
        
        layer_wise_distances: list[np.ndarray] = []
        for i, hidden_state in enumerate(search_hidden_states):
            if self.use_only_last_layer:
                samples = self.layer_wise_embeddings[self.NUM_HIDDEN_LAYERS - 1]
            else:
                samples = self.layer_wise_embeddings[i]
                
            index = self.build_index(samples)
            
            if self.feature_normalization:
                distances, indices = index.search(self.normalizer(hidden_state), self.topk)
            else:
                distances, indices = index.search(hidden_state, self.topk)
            layer_wise_distances.append(distances)

        # averaging top-k
        distances_score = rearrange(np.asarray(layer_wise_distances), "n b k -> (n b) k").mean(-1)
        reshaped_distances_score = rearrange(distances_score, "(b n) -> b n", n=num_layers) # [bl, n]
        
        # [b, ]
        temperatures: np.ndarray = np.sum(reshaped_distances_score * np.asarray(weights[:-1]), -1) + weights[-1]
        
        dst_logits = logits / np.expand_dims(temperatures, 1)
        
        return dst_logits
    
    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test
        
    def ll_lf(
        self,
        weights: np.ndarray,
        logits: np.ndarray,
        hidden_states_list: np.ndarray, 
        labels: np.ndarray, 
    ) -> np.ndarray:
        ## find optimal temperature with Cross-Entropy loss function
        
        logits = self.knn_weighting_numpy(logits, hidden_states_list, weights)
        probs = np_softmax(logits)
        onehot_labels = np.eye(probs.shape[-1])[labels]
        N = probs.shape[0]
        ce: np.ndarray = -np.sum(onehot_labels * np.log(probs + 1e-12)) / N
        return ce
    
    def optimize_dac_parameters_numpy(self, val_loader: torch.utils.data.DataLoader) -> list[float]:
        logits_list = []
        hidden_states_list = []
        labels_list = []
        device = self.deberta.device
        
        # cache logits and labels.
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = prepare_input(batch, device)
                logits, hidden_states = self.forward_logits(**batch)
                logits_list.append(logits.detach().cpu())
                if self.use_only_last_layer:
                    hidden_states = [hidden_states[-1]]
                    
                hidden_states_list.append(torch.stack(hidden_states, 2)[:, 0, :].detach().cpu())  # .mean(1)
                labels_list.append(batch["labels"].detach().cpu())
        
        logits = torch.cat(logits_list).cpu().numpy()
        hidden_states_list = torch.cat(hidden_states_list).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        
        st = time.time()
        params = optimize.minimize(
            self.ll_lf,
            self.init,
            args=(logits, hidden_states_list, labels),
            method="L-BFGS-B",
            bounds=self.bnds,
            tol=self.tol,
            options={"eps": self.eps, "disp": self.disp, "maxiter": 500},
        )
        ed = time.time()

        print("DAC Optimization done!: ({} sec)".format(ed - st))
        
        self.weights = params.x

        return self.weights
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=False if self.training else True,  # output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            
            if self.is_test:
                if self.use_only_last_layer:
                    hidden_states = [outputs.hidden_states[-1]]
                else:
                    hidden_states = outputs.hidden_states
                
                logits = torch.from_numpy(
                    self.knn_weighting_numpy(
                        logits=logits.detach().cpu().numpy(),
                        hidden_states=torch.stack(hidden_states, 2)[:, 0, :, :].detach().cpu().numpy(),  # .mean(1),
                        weights=self.weights
                    )
                ).to(self.deberta.device)

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)

            confidences = F.softmax(logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=pooled_output,
        )


class MDSNDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        self.class_cond_centroids = None
        self.class_cond_covariance = None
        
        self.is_test = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test
        
    def fit_ue(self, val_loader: torch.utils.data.DataLoader) -> None:
        print(f"Fitting {self.__class__.__name__} for Uncertainty Estimation...")
        batches = tqdm(val_loader)
        full_X_features = []
        full_labels = []
        for batch in batches:
            batch = prepare_input(batch)
            X_features = self.forward_features(**batch)
            labels = batch["labels"]
            full_X_features.append(X_features.detach().cpu().numpy())
            full_labels.append(labels.detach().cpu().numpy())
            
        full_X_features = np.concatenate(full_X_features, axis=0)
        full_labels = np.concatenate(full_labels, axis=0)
        
        self.class_cond_centroids = self._fit_centroids(full_X_features, full_labels)
        self.class_cond_covariance = self._fit_covariance(full_X_features, full_labels)
        print(f"Fitted {self.__class__.__name__} for Uncertainty Estimation.")
    
    def _fit_covariance(self, X, y, class_cond=True):
        if class_cond:
            return compute_covariance(self.class_cond_centroids, X, y, class_cond)
        return compute_covariance(self.train_centroid, X, y, class_cond)
        
    def _fit_centroids(self, X, y, class_cond=True):
        return compute_centroids(X, y, class_cond)
            
    def forward_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        # logits: torch.Tensor = self.classifier(pooled_output)


        loss = None
        losses = None
        confidences = None
        logits = None
        md = None
        if labels is not None:
            if self.is_test:
                md, inf_time = mahalanobis_distance(
                    train_features=None,
                    train_labels=None,
                    eval_features=pooled_output.detach().cpu().numpy(),
                    centroids=self.class_cond_centroids,
                    covariance=self.class_cond_covariance,
                )
            
            logits: torch.Tensor = self.classifier(pooled_output)
            
            confidences = F.softmax(logits, dim=-1)
            
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            uncertainty_scores=md,
        )


class NUQDeBERTaV3ForSequenceClassification(
    BaselineDeBERTaV3ForSequenceClassification
):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)
        
        self.nuq_classifier = NuqClassifier(
            tune_bandwidth="classification",
            n_neighbors=32,
            log_pN=0.0,
        )
        
        self.is_test = False

        # Initialize weights and apply final processing
        self.post_init()
        
    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test
        
    def fit_ue(self, val_loader: torch.utils.data.DataLoader) -> None:
        print(f"Fitting {self.__class__.__name__} for Uncertainty Estimation...")
        batches = tqdm(val_loader)
        full_X_features = []
        full_labels = []
        for batch in batches:
            batch = prepare_input(batch)
            X_features = self.forward_features(**batch)
            labels = batch["labels"]
            full_X_features.append(X_features.detach().cpu().numpy())
            full_labels.append(labels.detach().cpu().numpy())
            
        full_X_features = np.concatenate(full_X_features, axis=0)
        full_labels = np.concatenate(full_labels, axis=0)
        
        self.nuq_classifier.fit(full_X_features, full_labels)
        print(f"Fitted {self.__class__.__name__} for Uncertainty Estimation.")
            
    def forward_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.Tensor:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        return pooled_output

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        tokens: Optional[list[list[str]]] = None,
        ner_tags: Optional[list[list[str]]] = None,
        word_ids: Optional[list[list[int]]] = None,
        valid_masks: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        masks_for_span: Optional[torch.Tensor] = None,
        labels_for_span: Optional[torch.Tensor] = None,
        binary_labels_for_span: Optional[torch.Tensor] = None,
        token_masks: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output: torch.Tensor = self.dropout(pooled_output)
        # logits: torch.Tensor = self.classifier(pooled_output)

        loss = None
        losses = None
        confidences = None
        logits = None
        uncertainty_scores = None
        if labels is not None:
            if self.is_test:
                X_features = pooled_output.detach().cpu().numpy()
                nuq_probs, log_epistemic_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features),
                                                                                  return_uncertainty="epistemic")
                _, log_aleatoric_uncs = self.nuq_classifier.predict_proba(np.asarray(X_features),
                                                                          return_uncertainty="aleatoric")
                
                uncertainty_scores = log_epistemic_uncs + log_aleatoric_uncs
            
            logits: torch.Tensor = self.classifier(pooled_output)
            
            confidences = F.softmax(logits, dim=-1)
            
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits, labels)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits, labels)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            uncertainty_scores=uncertainty_scores,
        )
