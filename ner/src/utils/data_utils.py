from dataclasses import dataclass
from typing import Any, Final, Optional, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)


def prepare_input(
    data: Union[torch.Tensor, Any],
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.to(device)
        else:
            data[k] = v

    return data


@dataclass
class MappedLabels:
    masks_for_span: torch.Tensor
    labels_for_span: torch.Tensor
    binary_labels_for_span: torch.Tensor


@dataclass
class CustomDataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    id2label: dict[int, str]
    label2id: dict[str, int]
    max_length: int
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    valid_mask_pad_id: int = 0
    masks_for_span_padding_id: int = 0
    labels_for_span_padding_id: int = 1
    binary_labels_for_span_padding_id: int = 0
    padding_side = "right"
    max_span_length: int = 5
    lowest_constraint_length: int = 100
    negative_subsample_probability: float = 0.1
    no_tokenizer_pad_keys: Final[tuple[str, ...]] = (
        "raw_tokens",
        "pieces",
        "offsets",
        "raw_labels",
        "tokens",
        "ner_tags",
        "word_ids",
    )

    def _extract_spans(
        self,
        labels: list[int],
        offsets: list[tuple[int, int]],
    ) -> Union[dict[int, dict[int, int]], None]:
        """return a tree-like structure to determine correct label for the current
        span under construction.
        """
        word_spans = []
        starts = None
        ends = None
        clabel = None

        for lidx, l in enumerate(labels):
            if l == 0 or l % 2 == 1:
                if clabel is not None:
                    ends = lidx - 1
                    word_spans.append((starts, ends, clabel))

                if l == 0:
                    starts = None
                    clabel = None
                else:
                    starts = lidx
                    clabel = l

        if clabel is not None:
            word_spans.append((starts, len(labels) - 1, clabel))

        # covert spans to offsets
        span_dict = {}

        for span in word_spans:
            starts = offsets[span[0]][0]
            ends = offsets[span[1]][-1]

            if ends >= self.max_length - 1:
                continue

            if starts not in span_dict:
                span_dict[starts] = {}

            span_dict[starts][offsets[span[1]][-1]] = span[2]

        return span_dict

    def label_mapping(
        self,
        labels: list[int],
        pieces: list[str],
        offsets: list[tuple[int, int]],
    ):
        assert len(offsets) == len(
            labels
        ), f"{labels} and doest not have the same length as {len(labels)} != {len(offsets)}"

        span_dict = self._extract_spans(labels=labels, offsets=offsets)
        if not span_dict:
            return None

        # generate a set of labels
        num_terms = min(self.max_span_length, len(pieces))
        b_counts = sum([True for key in self.label2id.keys() if key[:2] == "B-"])
        num_constraints = b_counts * (2 * len(pieces) - num_terms + 1) * num_terms // 2
        base = torch.ones((num_constraints, len(pieces), len(self.id2label)))
        # try to fill the base
        binary_labels = []
        current_constraint_id = 0
        for i in range(1, base.size(1) - 1, 1):
            # トークンiから最大スパン長だけfor文
            for j in range(i, min(base.size(1) - 1, i + self.max_span_length), 1):

                # last indexのOタグを抜く
                for l in range(len(self.id2label) - 1, 2):
                    base[current_constraint_id, i : j + 1] = 0
                    base[current_constraint_id, i, l] = 1
                    if j > i:
                        base[current_constraint_id, i + 1 : j + 1, l + 1] = 1
                    if j + 1 < base.size(1):
                        base[current_constraint_id, j + 1, l + 1] = 0

                    current_constraint_id += 1
                    # check whether this label is correct

                    if i in span_dict and j in span_dict[i] and span_dict[i][j] == l:
                        binary_labels.append(1)
                    else:
                        binary_labels.append(0)

        # we should not subsample the dataset because it will change the base-rate
        binary_labels = torch.tensor(
            binary_labels + [0] * (num_constraints - len(binary_labels))
        )
        masks = torch.tensor(
            [1] * current_constraint_id
            + [0] * (num_constraints - current_constraint_id),
            dtype=torch.bool,
        )

        random_mask = (
            torch.bernoulli(
                torch.full(
                    size=(num_constraints,),
                    fill_value=self.negative_subsample_probability,
                    dtype=torch.float32,
                )
            )
            > 0.5
        )

        random_mask = torch.logical_or(random_mask, binary_labels.bool())
        masks = torch.logical_and(random_mask, masks)

        return MappedLabels(
            masks_for_span=masks,
            labels_for_span=base,
            binary_labels_for_span=binary_labels,
        )

    def __call__(self, features: list[dict[str, Any]]):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        valid_masks = [feature["valid_masks"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        # padding_side = self.tokenizer.padding_side
        if self.padding_side == "right":
            batch["labels"] = [
                label + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
            batch["valid_masks"] = [
                valid_mask
                + [self.valid_mask_pad_id] * (sequence_length - len(valid_mask))
                for valid_mask in valid_masks
            ]
        else:
            batch["labels"] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + label
                for label in labels
            ]
            batch["valid_masks"] = [
                [self.valid_mask_pad_id] * (sequence_length - len(valid_mask))
                + valid_mask
                for valid_mask in valid_masks
            ]
        batch = {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batch.items()
            if k not in self.no_tokenizer_pad_keys
        }
        batch["tokens"] = [feature["tokens"] for feature in features]
        batch["ner_tags"] = [feature["ner_tags"] for feature in features]

        return batch


@dataclass
class CustomDataCollatorForTokenClassificationForSoftLabel:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    id2label: dict[int, str]
    label2id: dict[str, int]
    max_length: int
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    valid_mask_pad_id: int = 0
    masks_for_span_padding_id: int = 0
    labels_for_span_padding_id: int = 1
    binary_labels_for_span_padding_id: int = 0
    padding_side = "right"
    max_span_length: int = 5
    lowest_constraint_length: int = 100
    negative_subsample_probability: float = 0.1
    no_tokenizer_pad_keys: Final[tuple[str, ...]] = (
        "labels",
        "raw_tokens",
        "pieces",
        "offsets",
        "raw_labels",
        "tokens",
        "ner_tags",
        "word_ids",
    )

    def _extract_spans(
        self,
        labels: list[int],
        offsets: list[tuple[int, int]],
    ) -> Union[dict[int, dict[int, int]], None]:
        """return a tree-like structure to determine correct label for the current
        span under construction.
        """
        word_spans = []
        starts = None
        ends = None
        clabel = None

        for lidx, l in enumerate(labels):
            if l == 0 or l % 2 == 1:
                if clabel is not None:
                    ends = lidx - 1
                    word_spans.append((starts, ends, clabel))

                if l == 0:
                    starts = None
                    clabel = None
                else:
                    starts = lidx
                    clabel = l

        if clabel is not None:
            word_spans.append((starts, len(labels) - 1, clabel))

        # covert spans to offsets
        span_dict = {}

        for span in word_spans:
            starts = offsets[span[0]][0]
            ends = offsets[span[1]][-1]

            if ends >= self.max_length - 1:
                continue

            if starts not in span_dict:
                span_dict[starts] = {}

            span_dict[starts][offsets[span[1]][-1]] = span[2]

        return span_dict

    def label_mapping(
        self,
        labels: list[int],
        pieces: list[str],
        offsets: list[tuple[int, int]],
    ):
        assert len(offsets) == len(
            labels
        ), f"{labels} and doest not have the same length as {len(labels)} != {len(offsets)}"

        span_dict = self._extract_spans(labels=labels, offsets=offsets)
        if not span_dict:
            return None

        # generate a set of labels
        num_terms = min(self.max_span_length, len(pieces))
        b_counts = sum([True for key in self.label2id.keys() if key[:2] == "B-"])
        num_constraints = b_counts * (2 * len(pieces) - num_terms + 1) * num_terms // 2
        base = torch.ones((num_constraints, len(pieces), len(self.id2label)))
        # try to fill the base
        binary_labels = []
        current_constraint_id = 0
        for i in range(1, base.size(1) - 1, 1):
            # トークンiから最大スパン長だけfor文
            for j in range(i, min(base.size(1) - 1, i + self.max_span_length), 1):

                # last indexのOタグを抜く
                for l in range(len(self.id2label) - 1, 2):
                    base[current_constraint_id, i : j + 1] = 0
                    base[current_constraint_id, i, l] = 1
                    if j > i:
                        base[current_constraint_id, i + 1 : j + 1, l + 1] = 1
                    if j + 1 < base.size(1):
                        base[current_constraint_id, j + 1, l + 1] = 0

                    current_constraint_id += 1
                    # check whether this label is correct

                    if i in span_dict and j in span_dict[i] and span_dict[i][j] == l:
                        binary_labels.append(1)
                    else:
                        binary_labels.append(0)

        # we should not subsample the dataset because it will change the base-rate
        binary_labels = torch.tensor(
            binary_labels + [0] * (num_constraints - len(binary_labels))
        )
        masks = torch.tensor(
            [1] * current_constraint_id
            + [0] * (num_constraints - current_constraint_id),
            dtype=torch.bool,
        )

        random_mask = (
            torch.bernoulli(
                torch.full(
                    size=(num_constraints,),
                    fill_value=self.negative_subsample_probability,
                    dtype=torch.float32,
                )
            )
            > 0.5
        )

        random_mask = torch.logical_or(random_mask, binary_labels.bool())
        masks = torch.logical_and(random_mask, masks)

        return MappedLabels(
            masks_for_span=masks,
            labels_for_span=base,
            binary_labels_for_span=binary_labels,
        )

    def __call__(self, features: list[dict[str, Any]]):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        valid_masks = [feature["valid_masks"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        # padding_side = self.tokenizer.padding_side
        if self.padding_side == "right":
            soft_labels = [
                label
                + [[self.label_pad_token_id] * len(self.label2id)]
                * (sequence_length - len(label))
                for label in labels
            ]
            batch["valid_masks"] = [
                valid_mask
                + [self.valid_mask_pad_id] * (sequence_length - len(valid_mask))
                for valid_mask in valid_masks
            ]

        else:
            soft_labels = [
                [[self.label_pad_token_id] * len(self.label2id)]
                * (sequence_length - len(label))
                + label
                for label in labels
            ]
            batch["valid_masks"] = [
                [self.valid_mask_pad_id] * (sequence_length - len(valid_mask))
                + valid_mask
                for valid_mask in valid_masks
            ]

        batch = {
            k: torch.tensor(v, dtype=torch.int64)
            for k, v in batch.items()
            if k not in self.no_tokenizer_pad_keys
        }
        batch["labels"] = torch.tensor(soft_labels, dtype=torch.float)

        return batch
