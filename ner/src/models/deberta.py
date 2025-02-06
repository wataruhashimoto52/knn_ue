import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final, Optional, Tuple, Union

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from pydantic import BaseModel, Field
from scipy import optimize
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import (
    DebertaV2ForTokenClassification,
    DebertaV2Model,
    PretrainedConfig,
)

from algorithms.density_estimators import DensityEstimatorWrapper, NormalizingFlow
from algorithms.evidential_nn import BeliefMatchingLoss
from algorithms.label_smoothing import LabelSmoother
from algorithms.spectral_modules import spectral_norm_fc
from algorithms.evidential_worker import Tagger_Evidence, get_tagger_one_hot
from utils.data_utils import prepare_input
from utils.schemas import TokenClassifierOutputConf
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


@dataclass
class kNNUEIntermediateOutput:
    logits: np.ndarray
    distances: np.ndarray
    indices: np.ndarray
    distance_term: np.ndarray
    label_term: Optional[np.ndarray]
    neighbor_scores: Optional[np.ndarray]


class CustomizedOneHotEncoder:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.id2label = id2label
        self.encoder = OneHotEncoder(handle_unknown="ignore", categories=[[i for i in self.id2label]])
    
    def fit_transform(self, labels: torch.Tensor) -> torch.Tensor:
        labels: np.ndarray = labels.cpu().numpy().reshape(-1, 1)
        output = self.encoder.fit_transform(labels)
        return torch.from_numpy(output.toarray())


def labels_to_onehot_with_ignore(labels: torch.Tensor, num_classes: int):
    # One-HOT表現の初期化
    one_hot = torch.zeros((labels.shape[0], num_classes))
    
    for i, label in enumerate(labels):
        if label == -100:
            one_hot[i] = torch.zeros(num_classes)  # ラベルが-100の場合、その行をすべて0にする
        else:
            one_hot[i, label] = 1  # One-HOT表現に変換
    
    return one_hot


@dataclass
class kNNParameters:
    alpha: float
    temperature: float
    lamb: float
    bias: float


@dataclass
class DACParameters:
    weights: np.ndarray
    bias: float


class CategoricalData(BaseModel):
    global_cat_list: list[int] = Field(..., description="ラベルごとの出現回数")
    global_cat_dict: dict[str, int] = Field(..., description="ラベル名をキー、出現回数を値")
    global_cat_dis_list: list[float] = Field(
        ..., description="`gl_cat_list`を出現回数の総計で正規化"
    )
    # token_cat_list_dict: dict[str, list[int]] = Field(..., description="トークンごとにラベルごとの出現回数をカウント")
    # token_cat_dis_list_dict: dict[str, list[float]] = Field(..., description="トークンごとにラベルごとの出現回数をカウントし、正規化して分布を作る")


class IndexType(Enum):
    FLAT = auto()
    IVF = auto()
    PQ = auto()
    IVFPQ = auto()
    HNSW = auto()
    HNSW_PQ = auto()
    HNSW_IVFPQ = auto()


__budget_functions__ = {
    "one": lambda N: torch.ones_like(N),
    "log": lambda N: torch.log(N + 1.0),
    "id": lambda N: N,
    "id_normalized": lambda N: N / N.sum(),
    "exp": lambda N: torch.exp(N),
    "parametrized": lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device)),
}


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


def valid_sequence_output(
    sequence_output, valid_masks, attention_mask, labels: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    device = sequence_output.device
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(
        batch_size, max_len, feat_dim, dtype=torch.float32, device=device
    )
    valid_attention_mask = torch.zeros(
        batch_size, max_len, dtype=torch.long, device=device
    )

    valid_labels = None
    if labels is not None:
        valid_labels = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_masks[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]

                if labels is not None:
                    valid_labels[i][jj] = labels[i][j]

    return valid_output, valid_attention_mask, valid_labels


def remove_special_tokens(
    input_ids: torch.Tensor,
    emissions: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    batch_size, seq_length, dim = emissions.shape
    device = emissions.device
    valid_emissions = torch.zeros(batch_size, seq_length, dim).to(device)
    valid_attention_mask = torch.zeros(batch_size, seq_length).to(device)

    valid_labels = None
    if labels is not None:
        valid_labels = torch.zeros(batch_size, seq_length).to(device)

    for i in range(batch_size):
        ind = -1
        sep_index = int((input_ids[i] == 102).nonzero(as_tuple=True)[0])

        for j in range(1, sep_index):
            if int(labels[i][j]) == -100:
                continue

            ind += 1
            valid_emissions[i][ind] = emissions[i][j]
            valid_attention_mask[i][ind] = attention_mask[i][j]
            if labels is not None:
                valid_labels[i][ind] = labels[i][j]

    return valid_emissions, valid_attention_mask, valid_labels


def valid_sequence_output_soft_label_deberta(
    sequence_output,
    valid_masks,
    attention_mask,
    num_labels: int,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    device = sequence_output.device
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(
        batch_size, max_len, feat_dim, dtype=torch.float32, device=device
    )
    valid_attention_mask = torch.zeros(
        batch_size, max_len, dtype=torch.long, device=device
    )

    valid_labels = None
    if labels is not None:
        valid_labels = torch.zeros(
            batch_size, max_len, num_labels, dtype=torch.long, device=device
        )

    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_masks[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]

                if labels is not None:
                    valid_labels[i][jj] = labels[i][j]

    return valid_output, valid_attention_mask, valid_labels


def remove_special_tokens_soft_label_deberta(
    input_ids: torch.Tensor,
    emissions: torch.Tensor,
    attention_mask: torch.Tensor,
    num_labels: int,
    labels: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

    batch_size, seq_length, dim = emissions.shape
    device = emissions.device
    valid_emissions = torch.zeros(batch_size, seq_length, dim).to(device)
    valid_attention_mask = torch.zeros(batch_size, seq_length).to(device)

    valid_labels = None
    if labels is not None:
        valid_labels = torch.zeros(batch_size, seq_length, num_labels).to(device)

    for i in range(batch_size):
        ind = -1
        sep_index = int((input_ids[i] == 2).nonzero(as_tuple=True)[0])

        for j in range(1, sep_index):
            if -100 in labels[i][j].tolist():
                continue

            ind += 1
            valid_emissions[i][ind] = emissions[i][j]
            valid_attention_mask[i][ind] = attention_mask[i][j]
            if labels is not None:
                valid_labels[i][ind] = labels[i][j]

    return valid_emissions, valid_attention_mask, valid_labels


class BaselineDeBERTaV3ForTokenClassification(DebertaV2ForTokenClassification):
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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

            confidences = F.softmax(logits, dim=-1)
            logits_reshaped = rearrange(logits, "b l k -> (b l) k")
            labels_reshaped = rearrange(labels, "b l -> (b l)")

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(logits_reshaped, labels_reshaped)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(logits_reshaped, labels_reshaped)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=sequence_output,
        )


class TemperatureScaledDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(
        self, config: PretrainedConfig, device: torch.device, tau: float = 1.0
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.tau = tau * torch.ones(1).to(device)
        self.is_test = False

        # Initialize weights and apply final processing
        self.post_init()

    def temperature_scale(self, reshaped_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.tau.unsqueeze(1).expand(
            reshaped_logits.size(0), reshaped_logits.size(1)
        )
        return reshaped_logits / temperature

    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test

    def get_optimized_temperature(
        self, val_loader: torch.utils.data.DataLoader
    ) -> float:
        self.tau = nn.Parameter(self.tau)  # make learnable
        logits_list = []
        labels_list = []
        device = self.deberta.device
        with torch.no_grad():
            for batch in val_loader:
                batch = prepare_input(batch, device)
                logits = self.forward_logits(**batch)

                logits_list.append(rearrange(logits, "b l k -> (b l) k"))
                labels_list.append(rearrange(batch["labels"], "b l -> (b l)"))

        logits = torch.cat(logits_list).to(device)  # [B * L, K]
        labels = torch.cat(labels_list).to(device)  # [B * L, ]

        nll_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.LBFGS([self.tau], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss: torch.Tensor = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

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
    ) -> torch.Tensor:
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
        logits: torch.Tensor = self.classifier(sequence_output)

        return logits

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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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

        batch_size: int = logits.shape[0]
        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            reshaped_logits = rearrange(logits, "b l k -> (b l) k")
            reshaped_labels = rearrange(labels, "b l -> (b l)")

            if self.is_test:
                reshaped_logits = self.temperature_scale(reshaped_logits)

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(reshaped_logits, reshaped_labels)

            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(reshaped_logits, reshaped_labels)

            logits = rearrange(reshaped_logits, "(b l) k -> b l k", b=batch_size)
            confidences = F.softmax(logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class LabelSmoothingDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(self, config: PretrainedConfig, smoothing: float) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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
        logits: torch.Tensor = self.classifier(sequence_output)

        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            confidences = F.softmax(logits, dim=-1)
            reshaped_logits = rearrange(logits, "b l h -> (b l) h")
            reshaped_labels = rearrange(labels, "b l -> (b l)")
            losses: torch.Tensor = self.criterion_none(reshaped_logits, reshaped_labels)
            loss: torch.Tensor = self.criterion_mean(reshaped_logits, reshaped_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class MCDropoutDeBERTaV3ForTokenClassification(DebertaV2ForTokenClassification):
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class SpectralNormalizedDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class BeliefMatchingDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class DensitySoftmaxDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        density_estimator: DensityEstimatorWrapper,
        max_log_prob: float,
        id2labelcount: dict[int, int],
        feature_normalization: bool,
        device: torch.device,
        pca_model: Optional[faiss.VectorTransform] = None,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.density_estimator = density_estimator
        self.max_log_prob = max_log_prob
        self.id2labelcount = id2labelcount
        self.feature_normalization = feature_normalization
        self.device_type = device
        self.pca_model = pca_model

        # normalization
        total = sum(list(self.id2labelcount.values()))
        self.id2labelcount = {
            key: value / total for key, value in self.id2labelcount.items()
        }
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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
        reshaped_sequence_output: torch.Tensor = rearrange(
            sequence_output, "b l h -> (b l) h"
        )
        logits: torch.Tensor = self.classifier(sequence_output)

        batch_size: InterruptedError = logits.shape[0]
        loss = None
        losses = None
        confidences = None
        span_logits = None
        if labels is not None:
            with torch.inference_mode():
                if self.feature_normalization:
                    log_probs = self.density_estimator.log_prob(
                        self.normalizer(reshaped_sequence_output)
                    )
                else:
                    if self.pca_model:
                        log_probs = self.density_estimator.log_prob(
                            torch.from_numpy(
                                self.pca_model.apply(
                                    reshaped_sequence_output.detach().cpu().numpy()
                                )
                            ).to(logits.device)
                        )  # [B, ]
                    else:
                        log_probs = self.density_estimator.log_prob(
                            reshaped_sequence_output
                        )  # [B, ]
            normalized_probs = torch.exp(log_probs / self.max_log_prob)
            logits_reshaped = rearrange(
                logits, "b l k -> (b l) k"
            )  # logits.view(-1, self.num_labels)
            labels_reshaped = rearrange(labels, "b l -> (b l)")

            dst_logits: torch.Tensor = logits_reshaped * torch.sigmoid(
                normalized_probs.unsqueeze(1)
            )

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(dst_logits, labels_reshaped)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(dst_logits, labels_reshaped)

            # reconstruct and compute confidence
            dst_logits = rearrange(dst_logits, "(b l) k -> b l k", b=batch_size)
            confidences = F.softmax(dst_logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class AutoTuneFaissKNNDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        feature_normalization: bool,
        topk: int,
        alpha: float,
        use_pca: bool,
        use_opq: bool,
        use_recomp: bool,
        index_type: str,
        knn_temperature: int = 800,
        transform_dim: int = 256,
        nprobe: int = 16,
        n_list: int = 100,
        n_subvectors: int = 32,
        use_neighbor_labels: bool = False,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.alpha = alpha * torch.ones(1).to(device)
        self.knn_temperature = knn_temperature * torch.ones(1).to(device)
        self.feature_normalization = feature_normalization
        self.topk = topk
        self.use_opq = use_opq
        self.use_pca = use_pca
        self.nprobe = nprobe
        self.use_recomp = use_recomp
        self.index_type = index_type
        self.n_list = n_list
        self.n_subvectors = n_subvectors
        self.transform_dim = transform_dim
        self.device_type = device
        self.is_test = False
        self.use_neighbor_labels = use_neighbor_labels

        self.tol = 1e-12
        self.eps = 1e-7
        self.disp = False

        # scaling parameter, temperature parameter.
        if self.use_neighbor_labels:
            self.bnds = [[0, 3.0]] + [[0, 20.0]] + [[0, 3.0]] + [[-3.0, 3.0]]
            self.init = [1.0] + [1.0] + [1.0] + [1.0]
        else:
            self.bnds = [[0, 3.0]] + [[0, 20.0]]
            self.init = [1.0] + [1.0]

        self.weights: list[float] = []

        self.ll_lf_criterion = nn.CrossEntropyLoss()

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
                outputs: TokenClassifierOutputConf = self.forward(**batch)

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
            embeddings.extend(filtered_reshaped_sequence_output)
            cached_labels.append(
                np.array(
                    [l for l in reshaped_labels.detach().cpu().tolist() if l != -100]
                )
            )

        encoded_train_samples: np.ndarray = (
            torch.stack(embeddings).detach().cpu().numpy()
        )
        train_cached_labels: np.ndarray = np.concatenate(cached_labels)

        dim = (
            self.transform_dim
            if self.use_pca or self.use_opq
            else self.config.hidden_size
        )

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
            base_index = faiss.IndexIVFScalarQuantizer(
                quantizer, dim, self.n_list, faiss.ScalarQuantizer.QT_8bit
            )

        elif self.index_type == IndexType.IVFPQ.name:
            quantizer = faiss.IndexFlatL2(dim)
            # quantizer, dim, n_list, pq_subvec, nbit
            base_index = faiss.IndexIVFPQ(
                quantizer, dim, self.n_list, self.n_subvectors, 8
            )

        elif self.index_type == IndexType.HNSW_PQ.name:
            base_index = faiss.IndexHNSWPQ(dim, self.n_subvectors, 32)
        elif self.index_type == IndexType.HNSW_IVFPQ.name:
            quantizer = faiss.IndexHNSWFlat(self.transform_dim, 32)
            base_index = faiss.IndexIVFPQ(
                quantizer, self.transform_dim, self.n_list, self.n_subvectors, 8
            )
        else:
            raise NotImplementedError()

        if encoded_train_samples.dtype != np.float32:
            encoded_train_samples = encoded_train_samples.astype(np.float32)

        if self.feature_normalization:
            encoded_train_samples = self.normalizer(encoded_train_samples)

        # train pca.
        if self.use_opq:
            self.vtrans = faiss.OPQMatrix(
                self.config.hidden_size, M=self.n_subvectors, d2=self.transform_dim
            )
            self.vtrans.train(encoded_train_samples)
        elif self.use_pca:
            self.vtrans = faiss.PCAMatrix(self.config.hidden_size, self.transform_dim)
            self.vtrans.train(encoded_train_samples)
        else:
            self.vtrans = None

        if self.use_opq:
            if self.index_type == "IVFPQ":
                base_index = faiss.IndexPreTransform(self.vtrans, base_index)
            else:
                raise NotImplementedError()

        elif self.use_pca:
            base_index = faiss.IndexPreTransform(self.vtrans, base_index)

        if "IVF" in self.index_type:
            base_index.nprobe = self.nprobe

        if self.device_type.type == "cuda":
            if "PQ" in self.index_type:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, base_index, co)

            elif self.index_type == "IVF":
                co = faiss.GpuMultipleClonerOptions()
                co.useFloat16 = True
                co.useFloat16CoarseQuantizer = True
                co.shard = True
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
        self.encoded_train_samples = encoded_train_samples

    def set_knn_temperature(self, temperature: Union[int, float, torch.Tensor]) -> None:
        self.knn_temperature = float(temperature) * torch.ones(1).to(
            self.deberta.device
        )

    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test

    def optimize_knn_parameters(
        self, val_loader: torch.utils.data.DataLoader
    ) -> kNNParameters:
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
                logits_list.append(rearrange(logits, "b l k -> (b l) k"))
                pooled_output_list.append(
                    rearrange(sequence_output, "b l h -> (b l) h")
                )
                labels_list.append(rearrange(batch["labels"], "b l -> (b l)"))

        logits = torch.cat(logits_list).to(device)
        pooled_outputs = torch.cat(pooled_output_list).to(device)
        labels = torch.cat(labels_list).to(device)

        nll_criterion = nn.CrossEntropyLoss()

        # optimize knn temperature.
        temperatures = list(range(500, 2600, 100))
        temp2loss: dict[int, float] = {}
        for temp in temperatures:
            self.set_knn_temperature(temp)
            val_loss: torch.Tensor = nll_criterion(
                self.knn_weighting(logits, pooled_outputs), labels
            )
            temp2loss[temp] = float(val_loss.item())

        final_temperature: float = min(temp2loss, key=temp2loss.get)
        self.set_knn_temperature(final_temperature)

        # next, optimize alpha.
        self.alpha = nn.Parameter(self.alpha)
        optimizer = torch.optim.LBFGS([self.alpha], lr=0.01, max_iter=100)

        def eval():
            optimizer.zero_grad()
            loss: torch.Tensor = nll_criterion(
                self.knn_weighting(logits, pooled_outputs), labels
            )
            loss.backward()
            return loss

        optimizer.step(eval)

        return kNNParameters(
            alpha=float(self.alpha),
            temperature=float(self.knn_temperature),
        )

    def ll_lf(
        self,
        weights: np.ndarray,
        logits: np.ndarray,
        hidden_states_list: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        ## find optimal temperature with Cross-Entropy loss function
        logits, _, _ = self.knn_weighting_numpy(logits, hidden_states_list, weights)
        probs = np_softmax(logits)
        filtered_probs = np.asarray([p for p, l in zip(probs, labels) if l != -100])
        filtered_labels = np.asarray([l for l in labels if l != -100])
        N = filtered_probs.shape[0]
        onehot_labels = np.eye(filtered_probs.shape[-1])[filtered_labels]

        loss: float = -np.sum(onehot_labels * np.log(filtered_probs + 1e-12)) / N
        return loss

    def optimize_knn_parameters_numpy(
        self, val_loader: torch.utils.data.DataLoader
    ) -> kNNParameters:
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
                logits_list.append(
                    rearrange(logits, "b l k -> (b l) k").detach().cpu().numpy()
                )
                pooled_output_list.append(
                    rearrange(sequence_output, "b l h -> (b l) h")
                    .detach()
                    .cpu()
                    .numpy()
                )
                labels_list.append(
                    rearrange(batch["labels"], "b l -> (b l)").detach().cpu().numpy()
                )

        logits = np.concatenate(logits_list)
        pooled_outputs = np.concatenate(pooled_output_list)
        labels = np.concatenate(labels_list)

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

        print("kNN-weighted Softmax Optimization done!: ({} sec)".format(ed - st))

        self.weights = params.x

        return kNNParameters(
            alpha=float(self.weights[0]),
            temperature=float(self.weights[1]),
            lamb=float(self.weights[2]),
            bias=float(self.weights[3]),
        )

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        logits: torch.Tensor = self.classifier(sequence_output)

        # [B, L, K], [B, L, D]
        return logits, sequence_output

    def knn_weighting(
        self, reshaped_logits: torch.Tensor, reshaped_sequence_output: torch.Tensor
    ) -> torch.Tensor:
        if self.feature_normalization:
            D, I = self.index.search(
                self.normalizer(reshaped_sequence_output).detach().cpu().numpy(),
                self.topk,
            )
        else:
            D, I = self.index.search(
                reshaped_sequence_output.detach().cpu().numpy(), self.topk
            )

        if self.use_recomp:
            knn_weight = self.alpha * torch.exp(
                -(
                    torch.cdist(
                        torch.from_numpy(self.encoded_train_samples[I]).to(
                            reshaped_logits.device
                        ),
                        reshaped_sequence_output.unsqueeze(1),
                        p=2,
                    ).squeeze()
                    / self.knn_temperature
                ).mean(-1)
            )
        else:
            distances = (
                torch.from_numpy(D)
                .to(reshaped_logits.device)
                .divide(self.knn_temperature)
            )
            knn_weight = self.alpha * torch.exp(-distances).mean(-1).to(
                reshaped_logits.device
            )

        dst_logits: torch.Tensor = reshaped_logits * knn_weight.unsqueeze(1)

        return dst_logits

    def knn_weighting_numpy(
        self,
        reshaped_logits: np.ndarray,
        reshaped_sequence_output: np.ndarray,
        params: list[float],
    ) -> kNNUEIntermediateOutput:
        if reshaped_logits.dtype != np.float32:
            reshaped_logits = reshaped_logits.astype(np.float32)

        if self.feature_normalization:
            query_vector = self.normalizer(reshaped_sequence_output)
        else:
            query_vector = reshaped_sequence_output

        D, I = self.index.search(query_vector, self.topk)

        if self.use_recomp:
            reshaped_indices = rearrange(I, "b k -> (b k)")
            extracted_raw_vectors = rearrange(
                self.encoded_train_samples[reshaped_indices],
                "(k b) d -> b k d",
                b=reshaped_sequence_output.shape[0],
                d=reshaped_sequence_output.shape[-1],
            )

            dist = np.linalg.norm(
                np.expand_dims(reshaped_sequence_output, 0)
                - rearrange(
                    extracted_raw_vectors,
                    "b k d -> k b d",
                    b=reshaped_sequence_output.shape[0],
                    k=self.topk,
                ),
                axis=-1,
            )
            distances = rearrange(dist, "k b -> b k") / params[1]
        else:  # use quantized distance
            distances: np.ndarray = D / params[1]  # [b, k]

        # distance_term = params[0] * np.exp(-distances.mean(-1))  # in-mean
        distance_term = params[0] * np.exp(-distances).mean(-1)  # out-mean
        knn_weight = distance_term
        label_term = None
        nearest_scores = None

        if self.use_neighbor_labels:
            pred_labels = reshaped_logits.argmax(-1)  # [B, ]
            values = rearrange(I, "b k -> (b k)")

            neighbor_labels = rearrange(
                self.train_cached_labels[values],
                "(b k) -> b k",
                b=reshaped_logits.shape[0],
                k=self.topk,
            )  # [B, K]
            neighbor_scores = np.sum(
                neighbor_labels == pred_labels.reshape(-1, 1), axis=1
            )
            # neighbor_scores[ = ]np.array(
            #     [(pred_labels[bt] == neighbor_labels[bt]).sum() for bt in range(reshaped_logits.shape[0])]
            # )
            label_term = params[2] * (neighbor_scores / self.topk + params[3])
            knn_weight += label_term

        dst_logits = reshaped_logits * np.expand_dims(knn_weight, 1)

        # return dst_logits, D, I
        return kNNUEIntermediateOutput(
            logits=dst_logits,
            distances=D,
            indices=I,
            distance_term=distance_term,
            label_term=label_term,
            neighbor_scores=neighbor_scores,
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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
        sequence_output: torch.Tensor = self.dropout(sequence_output)
        logits: torch.Tensor = self.classifier(sequence_output)

        batch_size: int = logits.shape[0]
        loss = None
        losses = None
        confidences = None
        span_logits = None

        reshaped_logits = rearrange(logits, "b l k -> (b l) k")
        reshaped_sequence_output = rearrange(sequence_output, "b l h -> (b l) h")
        distances = None
        indices = None
        if labels is not None:
            if self.is_test:
                # [B * L, K]
                """reshaped_logits = self.knn_weighting(
                    reshaped_logits=reshaped_logits,
                    reshaped_sequence_output=reshaped_sequence_output,
                )"""

                knnue_internal_output = self.knn_weighting_numpy(
                    reshaped_logits=reshaped_logits.detach().cpu().numpy(),
                    reshaped_sequence_output=reshaped_sequence_output.detach()
                    .cpu()
                    .numpy(),
                    params=self.weights,
                )

                reshaped_logits = torch.from_numpy(knnue_internal_output.logits).to(self.deberta.device)

            labels_reshaped = rearrange(labels, "b l -> (b l)")
            
            distances = knnue_internal_output.distances
            indices = knnue_internal_output.indices

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(reshaped_logits, labels_reshaped)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(reshaped_logits, labels_reshaped)

            # reconstruct and compute confidence
            logits = rearrange(reshaped_logits, "(b l) k -> b l k", b=batch_size)
            confidences = F.softmax(logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=sequence_output,
            distances=distances,
            indices=indices,
        )


class AutoTuneDACDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    M: Final[int] = 32  # 64, 32, 8
    NUM_HIDDEN_LAYERS: Final[int] = 13

    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
        feature_normalization: bool,
        use_only_last_layer: bool,
        topk: int,
        use_pca: bool,
        use_recomp: bool,
        index_type: str,
        transform_dim: int = 256,
        nprobe: int = 16,
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.deberta = DebertaV2Model(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.feature_normalization = feature_normalization
        self.use_only_last_layer = use_only_last_layer
        self.topk = topk
        self.use_pca = use_pca
        self.nprobe = nprobe
        self.use_recomp = use_recomp
        self.index_type = index_type
        self.transform_dim = transform_dim
        self.device_type = device
        self.is_test = False

        self.normalizer = lambda x: x / (
            torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )
        self.normalizer_numpy = lambda x: x / (
            np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )

        self.layer_wise_embeddings: dict[int, np.ndarray] = {}

        self.tol = 1e-3
        self.eps = 1e-7
        self.disp = False

        if use_only_last_layer:
            num_hidden_layer = 1
        else:
            num_hidden_layer = self.NUM_HIDDEN_LAYERS

        self.temperature_weights = torch.rand(num_hidden_layer).to(device)
        self.temperature_bias = torch.rand(1).to(device)

        self.bnds = [[0, 100.0]] * num_hidden_layer + [[-50.0, 50.0]]
        self.init = [1.0] * num_hidden_layer + [1.0]

        self.weights: list[float] = []

        self.ll_lf_criterion = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

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
            base_index = faiss.IndexIVFPQ(
                quantizer, self.transform_dim, self.config.num_labels, self.M, 8
            )
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

        if self.index_type == IndexType.HNSW_IVFPQ.name:
            base_index.nprobe = self.nprobe

        if self.device_type.type == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, base_index)

        elif self.device_type.type == "cpu":
            index = base_index
        else:
            raise NotImplementedError()

        index = base_index

        if not index.is_trained:
            index.train(samples)

        index.add(samples)

        return index

    def build_layer_wise_embeddings(
        self, train_dataloader: torch.utils.data.DataLoader
    ) -> None:
        embeddings = defaultdict(list)
        batches = tqdm(train_dataloader)
        device = self.deberta.device
        for batch in batches:
            batch = prepare_input(batch, device)
            with torch.inference_mode():
                outputs: TokenClassifierOutputConf = self.forward(**batch)

            reshaped_labels = batch["labels"].flatten()
            for i, hidden_state in enumerate(outputs.hidden_states):
                reshaped_hidden_state = torch.reshape(
                    hidden_state, (-1, hidden_state.shape[-1])
                )
                filtered_reshaped_hidden_state = torch.stack(
                    [
                        s
                        for s, l in zip(reshaped_hidden_state, reshaped_labels)
                        if l != -100
                    ]
                )
                embeddings[i].extend(filtered_reshaped_hidden_state)

        layer_wise_embeddings = {
            i: torch.stack(samples).detach().cpu().numpy()
            for i, samples in embeddings.items()
        }
        if self.feature_normalization:
            layer_wise_embeddings = {
                i: self.normalizer_numpy(samples)
                for i, samples in layer_wise_embeddings.items()
            }

        self.layer_wise_embeddings = layer_wise_embeddings

    def set_is_test(self, is_test: bool) -> None:
        self.is_test = is_test

    def ll_lf(
        self,
        weights: np.ndarray,
        logits: np.ndarray,
        hidden_states_list: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        ## find optimal temperature with Cross-Entropy loss function
        logits = self.knn_weighting_numpy(logits, hidden_states_list, weights)
        probs = np_softmax(logits)
        filtered_probs = np.asarray([p for p, l in zip(probs, labels) if l != -100])
        filtered_labels = np.asarray([l for l in labels if l != -100])
        N = filtered_probs.shape[0]
        onehot_labels = np.eye(filtered_probs.shape[-1])[filtered_labels]

        loss: float = -np.sum(onehot_labels * np.log(filtered_probs + 1e-12)) / N
        return loss

    def optimize_dac_parameters_numpy(
        self, val_loader: torch.utils.data.DataLoader
    ) -> list[float]:
        logits_list = []
        hidden_states_list = []
        labels_list = []
        device = self.deberta.device

        # cache logits and labels.
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = prepare_input(batch, device)
                # reshaped_logits: [B * L, K]
                # reshaped_sequence_output: [B * L, D]
                logits, hidden_states = self.forward_logits(**batch)
                logits_list.append(rearrange(logits, "b l k -> (b l) k"))
                if self.use_only_last_layer:
                    hidden_states = [hidden_states[-1]]
                hidden_states_list.append(
                    rearrange(torch.stack(hidden_states, 2), "b l n h -> (b l) n h")
                )
                labels_list.append(rearrange(batch["labels"], "b l -> (b l)"))

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
            options={"eps": self.eps, "disp": self.disp, "maxiter": 1200},
        )
        ed = time.time()

        w = params.x
        print("DAC Optimization done!: ({} sec)".format(ed - st))

        self.weights = w

        return self.weights

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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
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
            output_hidden_states=False
            if self.training
            else True,  # output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits: torch.Tensor = self.classifier(sequence_output)

        # [B, L, K], num_layer * [B, L, D]
        return logits, outputs.hidden_states

    def knn_weighting(
        self, reshaped_logits: torch.Tensor, reshaped_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """knn weighted temperature scaling

        Args:
            reshaped_logits (torch.Tensor): [B * L, K]
            reshaped_hidden_states (torch.Tensor) [B * L, N, H]
                B: batch size
                L: sequence length
                N: num of layers
                H: dim of hidden state

        Returns:
            torch.Tensor: _description_
        """
        batch_size_dot_length = reshaped_logits.shape[0]
        num_layers, dim_hidden_state = (
            reshaped_hidden_states.shape[1],
            reshaped_hidden_states.shape[2],
        )

        search_hidden_states = rearrange(reshaped_hidden_states, "bl n h -> n bl h")

        layer_wise_distances: list[np.ndarray] = []
        for i, hidden_state in enumerate(search_hidden_states):
            samples = self.layer_wise_embeddings[i]
            index = self.build_index(samples)
            if self.feature_normalization:
                distances, indices = index.search(
                    self.normalizer_numpy(hidden_state.detach().cpu().numpy()),
                    self.topk,
                )
            else:
                distances, indices = index.search(
                    hidden_state.detach().cpu().numpy(), self.topk
                )
            layer_wise_distances.append(distances)

        # distances: [(bn l) * k]

        # averaging top-k
        reshaped_distances = rearrange(
            torch.from_numpy(np.asarray(layer_wise_distances)).to(
                reshaped_logits.device
            ),
            "n bl k -> (n bl) k",
        )
        reshaped_distances_score = reshaped_distances.mean(-1)
        reshaped_distances_score = rearrange(
            reshaped_distances_score, "(bl n) -> bl n", n=num_layers
        )  # [bl, n]

        # [bl, ]
        plus_temperature_weights = F.softplus(self.temperature_weights)
        temperatures = (
            torch.sum(reshaped_distances_score * plus_temperature_weights, -1)
            + self.temperature_bias
        )

        print("temperatures", temperatures)

        dst_logits = reshaped_logits / temperatures.unsqueeze(1)

        return dst_logits

    def knn_weighting_numpy(
        self,
        reshaped_logits: np.ndarray,
        reshaped_hidden_states: np.ndarray,
        weights: list[float],
    ) -> np.ndarray:
        """knn weighted temperature scaling

        Args:
            reshaped_logits (torch.Tensor): [B * L, K]
            reshaped_hidden_states (torch.Tensor) [B * L, N, H]
                B: batch size
                L: sequence length
                N: num of layers
                H: dim of hidden state

        Returns:
            torch.Tensor: _description_
        """
        batch_size_dot_length = reshaped_logits.shape[0]
        num_layers, dim_hidden_state = (
            reshaped_hidden_states.shape[1],
            reshaped_hidden_states.shape[2],
        )

        search_hidden_states = rearrange(reshaped_hidden_states, "bl n h -> n bl h")
        if self.use_only_last_layer:
            num_layers = 1
            search_hidden_states: list[np.ndarray] = [search_hidden_states[-1]]

        layer_wise_distances: list[np.ndarray] = []
        for i, hidden_state in enumerate(search_hidden_states):
            if self.use_only_last_layer:
                samples = self.layer_wise_embeddings[self.NUM_HIDDEN_LAYERS - 1]
            else:
                samples = self.layer_wise_embeddings[i]

            index = self.build_index(samples)

            if self.feature_normalization:
                distances, indices = index.search(
                    self.normalizer_numpy(hidden_state), self.topk
                )
            else:
                distances, indices = index.search(hidden_state, self.topk)

            layer_wise_distances.append(distances)

        # distances: [(bn l) * k]

        # averaging top-k
        distances_score = rearrange(
            np.asarray(layer_wise_distances), "n bl k -> (n bl) k"
        ).mean(-1)
        reshaped_distances_score = rearrange(
            distances_score, "(bl n) -> bl n", n=num_layers
        )  # [bl, n]

        # [bl, ]
        temperatures: np.ndarray = (
            np.sum(reshaped_distances_score * np.asarray(weights[:-1]), -1)
            + weights[-1]
        )

        dst_logits = reshaped_logits / np.expand_dims(temperatures, 1)

        return dst_logits

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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
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
            output_hidden_states=False
            if self.training
            else True,  # output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits: torch.Tensor = self.classifier(sequence_output)

        batch_size: int = logits.shape[0]
        loss = None
        losses = None
        confidences = None
        span_logits = None

        reshaped_logits = rearrange(logits, "b l k -> (b l) k")
        if labels is not None:
            if self.is_test:
                # [B * L, K]
                reshaped_numpy_logits = self.knn_weighting_numpy(
                    reshaped_logits=reshaped_logits.detach().cpu().numpy(),
                    reshaped_hidden_states=rearrange(
                        torch.stack(outputs.hidden_states, 2), "b l n h -> (b l) n h"
                    )
                    .detach()
                    .cpu()
                    .numpy(),
                    weights=self.weights,
                )
                reshaped_logits = torch.from_numpy(reshaped_numpy_logits).to(
                    self.deberta.device
                )

            labels_reshaped = rearrange(labels, "b l -> (b l)")

            # compute loss
            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(reshaped_logits, labels_reshaped)
            loss_fnc_mean = CrossEntropyLoss()
            loss: torch.Tensor = loss_fnc_mean(reshaped_logits, labels_reshaped)

            # reconstruct and compute confidence
            logits = rearrange(reshaped_logits, "(b l) k -> b l k", b=batch_size)
            confidences = F.softmax(logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,  # (N, L, K)
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
            sequence_output=sequence_output,
        )
        
        
class EvidentialDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device,
    ) -> None:
        super().__init__(config)

        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels

        self.deberta = DebertaV2Model(config)

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        
        self.edl = Tagger_Evidence(
            num_classes=self.config.num_labels,
            device=device,
        )

        self.normalizer = lambda x: x / (
            torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )
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
        epoch: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        logits: torch.Tensor = self.classifier(sequence_output)
        reshaped_logits = rearrange(sequence_output, "b l k -> (b l) k")

        loss = None
        losses = None
        confidences = None
        span_logits = None
        batch_size = sequence_output.shape[0]

        if labels is not None:
            reshaped_labels = rearrange(labels, "b l -> (b l)")
            
            if self.training:
                loss, _ = self.edl.loss(
                    logits,
                    labels,
                    attention_mask,
                    epoch,
                )
            else:
                loss_fct = CrossEntropyLoss()
                loss: torch.Tensor = loss_fct(reshaped_logits, reshaped_labels)

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(reshaped_logits, reshaped_labels)

            confidences = F.softmax(logits, dim=-1)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )


class PosteriorNetworksDeBERTaV3ForTokenClassification(
    BaselineDeBERTaV3ForTokenClassification
):
    def __init__(
        self,
        config: PretrainedConfig,
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

        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, latent_dim)

        self.density_type = density_type
        self.feature_normalization = feature_normalization
        self.device_type = device
        self.pca_model = pca_model
        self.budget_function_initial = budget_function

        self.normalizer = lambda x: x / (
            torch.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10
        )
        self.regr = 1e-5
        
        self.customized_one_hot_encoder = CustomizedOneHotEncoder(
            id2label=self.config.id2label
        )

        if self.density_type in ("planar_flow", "radial_flow"):
            self.density_estimation = nn.ModuleList(
                [
                    NormalizingFlow(
                        dim=latent_dim, flow_length=n_density, flow_type=density_type
                    )
                    for c in range(self.config.num_labels)
                ]
            )
        else:
            raise NotImplementedError

        # Initialize weights and apply final processing

        self.categorical_data = None
        self.N = None
        self.budget_function = None
        self.post_init()
        
        
    def set_categorical_data(self, categorical_data: CategoricalData) -> None:
        self.categorical_data = categorical_data
        if self.budget_function_initial in __budget_functions__:
            self.N, self.budget_function = (
                __budget_functions__[self.budget_function_initial](self.categorical_data.global_cat_list),
                self.budget_function_initial,
            )
        else:
            raise NotImplementedError

    def compute_uce_loss(
        self,
        alpha: torch.Tensor,
        sequence_labels: torch.Tensor,
        input_mask: torch.Tensor,
        reduce: str = "sum"
    ) -> torch.Tensor:
        """_summary_

        Args:
            alpha (torch.Tensor): [B * L, O]
            sequence_labels (torch.Tensor): [B, L]
            input_mask: (torch.Tensor): [B, L]
            reduce (str, optional): _description_. Defaults to "sum".

        Returns:
            torch.Tensor: _description_
        """
        # with torch.autograd.detect_anomaly():
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.config.num_labels)
        entropy_reg = torch.distributions.Dirichlet(alpha).entropy()
        soft_output = get_tagger_one_hot(sequence_labels, self.config.num_labels, 0, 1,
                                         input_mask, self.deberta.device, is_2d=True)
        if reduce == "sum":
            ce_section = soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
            reshaped_input_mask = rearrange(input_mask, "b l -> (b l)").bool()
            expanded_input_mask = reshaped_input_mask.unsqueeze(-1).expand(-1, self.config.num_labels)
            ce_section = torch.masked_select(ce_section, expanded_input_mask)
            UCE_loss = torch.sum(ce_section) - self.regr * torch.sum(torch.masked_select(entropy_reg, reshaped_input_mask))
        
        elif reduce == "mean":
            ce_section = soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
            reshaped_input_mask = rearrange(input_mask, "b l -> (b l)").bool()
            expanded_input_mask = reshaped_input_mask.unsqueeze(-1).expand(-1, self.config.num_labels)
            ce_section = torch.masked_select(ce_section, expanded_input_mask)
            UCE_loss = torch.mean(ce_section) - self.regr * torch.mean(torch.masked_select(entropy_reg, reshaped_input_mask))
        else:
            UCE_loss = (
                soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
            ) - self.regr * entropy_reg
            
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
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutputConf]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        sequence_output: torch.Tensor = self.classifier(sequence_output)
        reshaped_sequence_output = rearrange(sequence_output, "b l o -> (b l) o")

        loss = None
        losses = None
        confidences = None
        span_logits = None
        batch_size = sequence_output.shape[0]
        seq_len = sequence_output.shape[1]

        if self.budget_function == "parametrized":
            N = self.N / self.N.sum()
        else:
            N = self.N

        if labels is not None:
            reshaped_labels = rearrange(labels, "b l -> (b l)")
            if self.feature_normalization:
                query_representation: torch.Tensor = self.normalizer(reshaped_sequence_output)
            else:
                query_representation: torch.Tensor = reshaped_sequence_output

            log_q_zk = torch.zeros((batch_size * seq_len, self.config.num_labels)).to(self.deberta.device)
            alpha = torch.zeros((batch_size * seq_len, self.config.num_labels)).to(self.deberta.device)
            
            for c in range(self.config.num_labels):
                log_probs = self.density_estimation[c].log_prob(query_representation)
                log_q_zk[:, c] = log_probs
                alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))

            dst_logits = torch.nn.functional.normalize(alpha, p=1)  # [B * L, K]
            loss: torch.Tensor = self.compute_uce_loss(
                alpha=alpha,
                sequence_labels=labels,
                input_mask=attention_mask,
                reduce="mean",
            )
            
            loss_fct = CrossEntropyLoss()
            loss: torch.Tensor = loss_fct(dst_logits, reshaped_labels)

            loss_fct = CrossEntropyLoss(reduction="none")
            losses: torch.Tensor = loss_fct(reshaped_sequence_output, reshaped_labels)

            confidences = F.softmax(rearrange(dst_logits, "(b l) k -> b l k", b=batch_size), dim=-1)

        if not return_dict:
            output = (dst_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutputConf(
            loss=loss,  # scalar
            logits=dst_logits,
            hidden_states=outputs.hidden_states,
            attentions=None,
            confidences=confidences,  # (N, L, K)
            losses=losses,  # (N, )
            span_logits=span_logits,
        )
