import itertools
import json
import logging
import os
import time
from typing import Optional, Union

import numpy as np
import regex as re
import torch
import torch.nn as nn
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers.trainer_pt_utils import nested_concat

from .data_utils import prepare_input
from .metrics_auprc import (
    compute_auprc_ner_flatten,
    compute_auprc_ner_flatten_span,
    compute_auprc_ner_flatten_without_other,
    compute_auprc_ner_sentence,
    compute_auprc_ner_sentence_span,
    compute_auprc_ner_sentence_without_other,
)
from .metrics_aurcs import (
    compute_aurc_eaurc_ner_flatten,
    compute_aurc_eaurc_ner_flatten_span,
    compute_aurc_eaurc_ner_flatten_without_other,
    compute_aurc_eaurc_ner_sentence,
    compute_aurc_eaurc_ner_sentence_span,
    compute_aurc_eaurc_ner_sentence_without_other,
)
from .metrics_calibration_errors import (
    compute_calibration_error_ner_flatten,
    compute_calibration_error_ner_flatten_span,
    compute_calibration_error_ner_flatten_without_other,
    compute_calibration_error_ner_sentence,
    compute_calibration_error_ner_sentence_span,
    compute_calibration_error_ner_sentence_without_other,
    save_sentence_span_probabilities,
)
from .metrics_rocauc import (
    compute_roc_auc_ner_flatten,
    compute_roc_auc_ner_flatten_span,
    compute_roc_auc_ner_flatten_without_other,
    compute_roc_auc_ner_sentence,
    compute_roc_auc_ner_sentence_span,
    compute_roc_auc_ner_sentence_without_other,
)
from .schemas import EvalPredictionV2, TokenClassifierOutputConf
from .train_utils import apply_dropout

logger = logging.getLogger(__name__)


def nested_numpify(tensors: torch.Tensor):
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)

    if tensors.requires_grad:
        return t.detach().numpy()
    return t.numpy()


def expected_calibration_errors(
    predictions: np.ndarray, confidences: np.ndarray, labels: np.ndarray, bins: int
) -> float:
    """

    Args:
        predictions (np.ndarray): predicted labels
        confidences (np.ndarray): confidences
        labels (np.ndarray): ground truth labels
        bins (int): number of bins

    Returns:
        _type_: _description_
    """

    conf = confidences
    # Storage
    acc_tab = np.zeros(bins)  # empirical (true) confidence
    mean_conf = np.zeros(bins)  # predicted confidence
    nb_items_bin = np.zeros(bins)  # number of items in the bins
    tau_tab = np.linspace(0, 1, bins + 1)  # confidence bins
    for i in np.arange(bins):  # iterate over the bins
        # select the items where the predicted max probability falls in the bin
        # [tau_tab[i], tau_tab[i + 1)]
        sec = (tau_tab[i + 1] > conf) & (conf >= tau_tab[i])
        nb_items_bin[i] = np.sum(sec)  # Number of items in the bin
        # select the predicted classes, and the true classes
        class_pred_sec, y_sec = predictions[sec], labels[sec]
        # average of the predicted max probabilities
        mean_conf[i] = np.mean(conf[sec]) if nb_items_bin[i] > 0 else np.nan
        # compute the empirical confidence
        acc_tab[i] = np.mean(class_pred_sec == y_sec) if nb_items_bin[i] > 0 else np.nan

    # Cleaning
    mean_conf = mean_conf[nb_items_bin > 0]
    acc_tab = acc_tab[nb_items_bin > 0]
    nb_items_bin = nb_items_bin[nb_items_bin > 0]

    # Reliability diagram
    reliability_diag = (mean_conf, acc_tab)
    # Expected Calibration Error
    ece = np.average(
        np.absolute(mean_conf - acc_tab),
        weights=nb_items_bin.astype(np.float) / np.sum(nb_items_bin),
    )
    # Maximum Calibration Error
    mce = np.max(np.absolute(mean_conf - acc_tab))
    # Saving
    cal = {"reliability_diag": reliability_diag, "ece": ece, "mce": mce}
    return cal


def binary_ECE(probs, y_true, power=1, bins=15):

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1
    bin_func = (
        lambda p, y, idx: (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power)
        * np.sum(idx)
        / len(probs)
    )

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece


def classwise_ECE(
    probs: np.ndarray,
    y_true: np.ndarray,
    power: int = 1,
    bins: int = 15,
    ignore_index: Optional[int] = None,
):
    """

    Args:
        probs (_type_): confidences (length, num_labels)
        y_true (_type_): labels (length, )
        power (int, optional): _description_. Defaults to 1.
        bins (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.mean(
        [
            binary_ECE(probs[:, c], y_true[:, c].astype(float), power=power, bins=bins)
            for c in range(n_classes)
            if c != ignore_index
        ]
    )


def compute_classwise_expected_calibration_error_ner_flatten(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
):
    """

    Args:
        evalprediction (EvalPredictionV2): _description_

    Returns:
        _type_: _description_
    """

    if evalprediction.tags:  # with CRF

        flatten_confidences = np.reshape(
            evalprediction.predictions, (-1, len(label_map))
        )
        flatten_label_ids = evalprediction.label_ids.flatten()
        processed_confidences = [
            c
            for (c, a) in zip(flatten_confidences, evalprediction.attentions.flatten())
            if int(a) == 1
        ]

    else:
        flatten_base_predictions = np.reshape(
            evalprediction.predictions, (-1, len(label_map))
        )
        flatten_confidences = flatten_base_predictions
        flatten_label_ids = evalprediction.label_ids.flatten()
        processed_confidences = [
            c for (c, l) in zip(flatten_confidences, flatten_label_ids) if l != -100
        ]
    processed_labels = [l for l in flatten_label_ids if l != -100]

    classwise_ece = classwise_ECE(
        probs=np.asarray(processed_confidences),
        y_true=np.asarray(processed_labels),
        bins=10,
        ignore_index=ignore_index,
    )

    return classwise_ece


def compute_brier_score_ner_flatten(evalprediction: EvalPredictionV2):
    """_summary_

    Args:
        evalprediction (EvalPredictionV2): _description_

    Returns:
        _type_: _description_
    """
    predictions = np.reshape(
        evalprediction.predictions, (-1, evalprediction.predictions.shape[-1])
    )
    label_ids = evalprediction.label_ids.flatten()

    if evalprediction.tags:
        processed_predictions = [
            p
            for (p, a) in zip(predictions, evalprediction.attentions.flatten())
            if int(a) == 1
        ]
    else:
        processed_predictions = [
            p for (p, l) in zip(predictions, label_ids) if l != -100
        ]
    processed_labels = [l for (p, l) in zip(predictions, label_ids) if l != -100]

    np_predictions = np.asarray(processed_predictions)
    np_labels = np.asarray(processed_labels)

    length = np_predictions.shape[0]
    final_predictions = np.asarray(
        [np_predictions[i, processed_labels[i]] for i in range(length)]
    )
    argmax_predictions = np.argmax(np_predictions, axis=1)
    final_labels = np.asarray(
        [1 if np_labels[i] == argmax_predictions[i] else 0 for i in range(length)]
    )

    brier_score = brier_score_loss(y_true=final_labels, y_prob=final_predictions)

    return brier_score


def compute_brier_score_ner(evalprediction: EvalPredictionV2):

    label_ids = evalprediction.label_ids
    brier_scores = []
    for i in range(evalprediction.predictions.shape[0]):

        if evalprediction.tags:  # with CRF
            pred = [
                p
                for (p, a) in zip(
                    evalprediction.predictions[i], evalprediction.attentions[i]
                )
                if int(a) == 1
            ]
        else:
            pred = [
                p
                for (p, l) in zip(evalprediction.predictions[i], label_ids[i])
                if l != -100
            ]

        label = [l for l in label_ids[i] if l != -100]

        pred = np.asarray(pred)
        label = np.asarray(label)

        final_pred = np.asarray([pred[i, label[i]] for i in range(pred.shape[0])])
        argmax_pred = np.argmax(pred, axis=1)

        final_label = np.asarray(
            [1 if label[i] == argmax_pred[i] else 0 for i in range(pred.shape[0])]
        )

        brier_score = brier_score_loss(y_true=final_label, y_prob=final_pred)
        brier_scores.append(brier_score)

    return np.mean(brier_scores)


def rcc_auc(conf: np.ndarray, risk: np.ndarray, return_points=False):
    """

    Args:
        conf (np.ndarray): confidences (length, )
        risk (np.ndarray): binary array
        return_points (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    n = len(conf)
    cr_pair = list(zip(conf, risk))
    cr_pair.sort(key=lambda x: x[0], reverse=True)

    cumulative_risk = [cr_pair[0][1]]
    for i in range(1, n):
        cumulative_risk.append(cr_pair[i][1] + cumulative_risk[-1])

    points_x = []
    points_y = []

    auc = 0
    for k in range(n):
        auc += cumulative_risk[k] / (1 + k)
        points_x.append((1 + k) / n)  # coverage
        points_y.append(cumulative_risk[k] / (1 + k))  # current avg. risk

    if return_points:
        return auc, points_x, points_y
    else:
        return auc


def get_aurc_eaurc(residuals: np.ndarray, confidence: np.ndarray):

    curve = []
    m = len(residuals)
    idx_sorted = np.argsort(confidence)
    temp1 = residuals[idx_sorted]
    cov = len(temp1)
    acc = sum(temp1)
    curve.append((cov / m, acc / len(temp1)))
    for i in range(0, len(idx_sorted) - 1):
        cov = cov - 1
        acc = acc - residuals[idx_sorted[i]]
        curve.append((cov / m, acc / (m - i)))
    aurc = sum([a[1] for a in curve]) / len(curve)
    err = np.mean(residuals)
    kappa_star_aurc = err + (1 - err) * (np.log(1 - err))
    eaurc = aurc - kappa_star_aurc
    return curve, aurc * 1000, eaurc * 1000


def compute_aurc_eaurc(evalprediction: EvalPredictionV2, level: str = "token") -> float:

    if level != "token":
        raise NotImplementedError

    if evalprediction.tags:  # with CRF
        predictions = (
            evalprediction.tags if evalprediction.tags else evalprediction.predictions
        )
        confidences = evalprediction.predictions

        final_predictions = np.asarray(list(itertools.chain.from_iterable(predictions)))
        final_labels = np.asarray(
            [l for l in evalprediction.label_ids.flatten() if int(l) != -100]
        )
        base_confidences = np.max(
            np.reshape(confidences, (-1, confidences.shape[-1])), axis=-1
        )
        base_labels = evalprediction.label_ids.flatten()  # (length, )
        final_confidences = np.asarray(
            [
                p
                for (p, a) in zip(base_confidences, evalprediction.attentions.flatten())
                if int(a) == 1
            ]
        )

    else:
        predictions = np.argmax(evalprediction.predictions, axis=2)
        confidences = evalprediction.predictions

        base_predictions = predictions.flatten()  # (length, )
        base_confidences = np.max(
            np.reshape(confidences, (-1, confidences.shape[-1])), axis=-1
        )  # (length, )
        base_labels = evalprediction.label_ids.flatten()  # (length, )

        final_predictions = np.asarray(
            [p for (p, l) in zip(base_predictions, base_labels) if l != -100]
        )
        final_labels = np.asarray(
            [l for (p, l) in zip(base_predictions, base_labels) if l != -100]
        )
        final_confidences = np.asarray(
            [p for (p, l) in zip(base_confidences, base_labels) if l != -100]
        )

    errors = (final_labels != final_predictions).astype(int)

    curve, aurc, eaurc = get_aurc_eaurc(residuals=errors, confidence=final_confidences)

    return aurc, eaurc


def compute_aurc_eaurc_for_span(
    evalprediction: EvalPredictionV2,
    level: str = "token",
    use_mean: bool = False,
    label_map: Optional[dict] = None,
) -> float:

    if level != "token":
        raise NotImplementedError

    if evalprediction.tags:  # with CRF
        # use tag from forward-backward algorithm's output confidence
        predictions = np.argmax(evalprediction.predictions, axis=2)  # (N, length)
        confidences = np.reshape(evalprediction.predictions, (-1, len(label_map)))

        base_predictions = predictions.flatten()  # (length, )
        base_labels = evalprediction.label_ids.flatten()  # (length, )

        base_confidences = np.asarray(
            [c for (c, l) in zip(confidences, base_labels) if l != -100]
        )

        processed_label_entities = get_entities(
            [label_map[l] for l in base_labels if l != -100]
        )
        processed_predicted_entities = get_entities(
            [label_map[p] for (p, l) in zip(base_predictions, base_labels) if l != -100]
        )

        processed_label_entities_set = set(processed_label_entities)

        processed_errors_span = [
            0 if predicted_entity in processed_label_entities_set else 1
            for predicted_entity in processed_predicted_entities
        ]
        if use_mean:
            processed_predictions_span = [
                np.mean(
                    np.max(
                        confidences[predicted_entity[1] : predicted_entity[2] + 1, :], 1
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]
        else:
            processed_predictions_span = [
                np.prod(
                    np.max(
                        confidences[predicted_entity[1] : predicted_entity[2] + 1, :], 1
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]

        if not processed_predicted_entities:
            return None, None

    else:
        predictions = np.argmax(evalprediction.predictions, axis=2)  # (N, length)
        confidences = np.reshape(evalprediction.predictions, (-1, len(label_map)))

        base_predictions = predictions.flatten()  # (length, )
        base_labels = evalprediction.label_ids.flatten()  # (length, )

        base_confidences = np.asarray(
            [c for (c, l) in zip(confidences, base_labels) if l != -100]
        )

        processed_label_entities = get_entities(
            [label_map[l] for l in base_labels if l != -100]
        )
        processed_predicted_entities = get_entities(
            [label_map[p] for (p, l) in zip(base_predictions, base_labels) if l != -100]
        )

        processed_label_entities_set = set(processed_label_entities)

        processed_errors_span = [
            0 if predicted_entity in processed_label_entities_set else 1
            for predicted_entity in processed_predicted_entities
        ]
        if use_mean:
            processed_predictions_span = [
                np.mean(
                    np.max(
                        base_confidences[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                        1,
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]
        else:
            processed_predictions_span = [
                np.prod(
                    np.max(
                        base_confidences[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                        1,
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]

        if not processed_predicted_entities:
            return None, None

    curve, aurc, eaurc = get_aurc_eaurc(
        residuals=np.asarray(processed_errors_span),
        confidence=np.asarray(processed_predictions_span),
    )

    return aurc, eaurc


def compute_aurc_eaurc_for_span_v2(
    evalprediction: EvalPredictionV2,
    level: str = "token",
    use_mean: bool = False,
    label_map: Optional[dict] = None,
) -> float:

    if level != "token":
        raise NotImplementedError

    # use tag from forward-backward algorithm's output confidence (CRF)
    # use softmax confidence (not-CRF)
    predictions = np.argmax(evalprediction.predictions, axis=2)  # (N, length)
    confidences = np.reshape(evalprediction.predictions, (-1, len(label_map)))

    base_predictions = predictions.flatten()  # (length, )
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    base_confidences = np.asarray(
        [c for (c, l) in zip(confidences, base_labels) if l != -100]
    )

    processed_label_entities = get_entities(
        [label_map[l] for l in base_labels if l != -100]
    )
    processed_predicted_entities = get_entities(
        [label_map[p] for (p, l) in zip(base_predictions, base_labels) if l != -100]
    )
    if not processed_predicted_entities:
        return None, None

    preds = []
    # trues = []
    errors = []
    for i in range(len(base_confidences)):
        e_true = [ent for ent in processed_label_entities if ent[1] == i]
        e_pred = [ent for ent in processed_predicted_entities if ent[1] == i]

        if not e_true and not e_pred:
            continue
        if e_pred:
            if use_mean:
                conf = np.mean(
                    np.max(base_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            else:
                conf = np.prod(
                    np.max(base_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            if not e_true:
                preds.append(conf)
                # trues.append(0)
                errors.append(1)
            elif e_true[0] == e_pred[0]:
                preds.append(conf)
                # trues.append(1)
                errors.append(0)
            else:
                preds.append(conf)
                # trues.append(0)
                errors.append(1)

        else:  # not e_pred
            if e_true:
                preds.append(0)
                # trues.append(1)
                errors.append(1)

            else:  # ここに来ることはない．
                preds.append(0)
                # trues.append(0)
                errors.append(0)

    curve, aurc, eaurc = get_aurc_eaurc(
        residuals=np.asarray(errors), confidence=np.asarray(preds)
    )

    return aurc, eaurc


def align_predictions(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    label_map: dict,
    tags: Optional[list[list[int]]] = None,
) -> tuple[list[int], list[int]]:

    if tags:
        preds = np.asarray(tags)
        batch_size = preds.shape[0]

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(label_ids.shape[1]):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
            preds_list[i].extend([label_map[tag] for tag in tags[i]])

    else:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def make_eval_prediction(
    model: Union[PreTrainedModel, list[PreTrainedModel]],
    dataloader: DataLoader,
    label_map: dict[int, str],
    output_path: str,
    seed: int,
    calibration_algorithm: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    num_monte_carlo: int = 20,
    datastore: Optional[list[dict]] = None,
    split: str = "test",
) -> EvalPredictionV2:
    all_losses = None
    all_preds = None
    all_labels = None
    all_attentions = None
    all_input_ids = None
    all_knn_distances = None
    all_knn_indices = None
    all_tags = []
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    start = 0
    # for Deep Ensemble
    if type(model) == list:
        batches = tqdm(dataloader)
        start = time.time()
        for step, data in enumerate(batches):
            if device:
                data = prepare_input(data, device)

            sub_all_preds = []
            sub_all_losses = []

            for sub_model in model:
                with torch.inference_mode():
                    output: TokenClassifierOutputConf = sub_model(**data)

                sub_predictions = output.confidences
                sub_losses = output.losses

                labels = nested_numpify(data["labels"])

                sub_all_losses.append(sub_losses)
                sub_all_preds.append(sub_predictions)

            labels = nested_numpify(data["labels"])
            predictions = nested_numpify(torch.stack(sub_all_preds, dim=-1).mean(-1))
            losses = torch.stack(sub_all_losses, dim=-1).mean(-1)
            attentions = None  # 現状はCRFをアンサンブルしていないため
            tags = None  # 現状はCRFをアンサンブルしていないため

            all_input_ids = (
                data["input_ids"]
                if all_input_ids is None
                else nested_concat(
                    all_input_ids,
                    data["input_ids"],
                    padding_index=tokenizer.pad_token_id,
                )
            )

            all_preds = (
                predictions
                if all_preds is None
                else nested_concat(all_preds, predictions, padding_index=-100)
            )
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

            if attentions is not None:
                attentions = nested_numpify(output.attentions)
                all_attentions = (
                    attentions
                    if all_attentions is None
                    else nested_concat(all_attentions, attentions, padding_index=0)
                )

            if losses is not None:
                losses = nested_numpify(losses)
                all_losses = (
                    losses
                    if all_losses is None
                    else nested_concat(all_losses, losses, padding_index=-100)
                )

            if tags is not None:
                all_tags.extend(output.tags)

    else:
        model.eval()
        batches = tqdm(dataloader)

        start = time.time()
        for step, data in enumerate(batches):
            if device:
                data = prepare_input(data, device)
            if "MCDropout" in model.__class__.__name__:

                # model.train()
                model.apply(apply_dropout)
                confidences = []
                loss = []
                tag = []
                for i in range(num_monte_carlo):
                    with torch.inference_mode():
                        output: TokenClassifierOutputConf = model(**data)
                    confidences.append(output.confidences.cpu())
                    loss.append(output.losses.cpu())
                    tag.append(output.tags)

                stacked_confidences = torch.stack(confidences, dim=-1)
                predictions = nested_numpify(stacked_confidences.mean(-1))

                stacked_loss = torch.stack(loss, dim=-1)
                losses = stacked_loss.mean(-1)
                tags = output.tags  # 一旦仮置。CRFを使うことになったら考える。
                attentions = output.attentions
                knn_distances = output.distances
                knn_indices = output.indices

            elif re.search(r"EntityDistance", model.__class__.__name__):
                data["datastore"] = datastore
                with torch.inference_mode():
                    output: TokenClassifierOutputConf = model(**data)

                predictions = nested_numpify(output.confidences)
                losses = output.losses
                tags = output.tags
                attentions = output.attentions
                knn_distances = output.distances
                knn_indices = output.indices

            else:
                with torch.inference_mode():
                    output: TokenClassifierOutputConf = model(**data)

                predictions = nested_numpify(output.confidences)
                losses = output.losses
                tags = output.tags
                attentions = output.attentions
                knn_distances = output.distances
                knn_indices = output.indices

            labels = nested_numpify(data["labels"])

            all_input_ids = (
                data["input_ids"]
                if all_input_ids is None
                else nested_concat(
                    all_input_ids,
                    data["input_ids"],
                    padding_index=tokenizer.pad_token_id,
                )
            )

            all_preds = (
                predictions
                if all_preds is None
                else nested_concat(all_preds, predictions, padding_index=-100)
            )
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

            all_attentions = (
                attentions
                if all_attentions is None
                else nested_concat(all_attentions, attentions, padding_index=0)
            )

            if attentions is not None:
                attentions = nested_numpify(output.attentions)
                all_attentions = (
                    attentions
                    if all_attentions is None
                    else nested_concat(all_attentions, attentions, padding_index=0)
                )

            if knn_distances is not None:
                all_knn_distances = (
                    knn_distances
                    if all_knn_distances is None
                    else nested_concat(
                        all_knn_distances, knn_distances, padding_index=0
                    )
                )

            if knn_indices is not None:
                all_knn_indices = (
                    knn_indices
                    if all_knn_indices is None
                    else nested_concat(all_knn_indices, knn_indices, padding_index=0)
                )

            if losses is not None:
                losses = nested_numpify(output.losses)
                all_losses = (
                    losses
                    if all_losses is None
                    else nested_concat(all_losses, losses, padding_index=-100)
                )

            if tags is not None:
                all_tags.extend(output.tags)

    end = time.time()

    inference_time = str(end - start)

    if split == "test":
        with open(
            os.path.join(output_path, f"test_inference_time_{str(seed)}.json"), "w"
        ) as f:
            json.dump(
                {
                    "inference_time": inference_time,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    if split == "test":
        records = []
        for j in range(all_preds.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[j].tolist(), skip_special_tokens=True
            )
            confs = all_preds[j].max(-1)
            preds = all_preds[j].argmax(-1)
            labs = all_labels[j]
            filtered_preds = [
                label_map[int(p)] for p, l in zip(preds, labs) if l != -100
            ]
            filtered_confs = [c.tolist() for c, l in zip(confs, labs) if l != -100]
            filtered_labs = [label_map[int(l)] for l in labs if l != -100]

            record_dict = {
                "tokens": [
                    comp for t in tokens if (comp := t.replace("▁", "")) and comp != ""
                ],
                "confidences": filtered_confs,
                "predictions": filtered_preds,
                "labels": filtered_labs,
            }
            if all_knn_distances is not None and all_knn_indices is not None:
                record_dict.update(
                    {
                        "knn_distances": all_knn_distances[j].tolist(),
                        "knn_indices": all_knn_indices[j].tolist(),
                    }
                )

            records.append(record_dict)

        with open(
            os.path.join(output_path, f"test_predictions_{str(seed)}.json"), "w"
        ) as f:
            json.dump(records, f, ensure_ascii=False, indent=4)

    return EvalPredictionV2(
        predictions=all_preds,
        label_ids=all_labels,
        losses=all_losses,
        tags=all_tags,
        attentions=all_attentions,
    )


def compute_metrics(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    output_path: str,
    split: str = "test",
    seed: int = 1,
) -> dict[str, float]:
    """compute each metrics.

    Args:
        evalprediction (EvalPredictionV2): Formatted Prediction
        label_map (dict): label id to label name dictionary

    Returns:
        dict[str, float]: each metrics scores
    """
    preds_list, out_label_list = align_predictions(
        evalprediction.predictions,
        evalprediction.label_ids,
        label_map=label_map,
        tags=evalprediction.tags,
    )

    aurc_flatten, eaurc_flatten = compute_aurc_eaurc_ner_flatten(
        evalprediction,
    )
    (
        aurc_flatten_without_other,
        eaurc_flatten_without_other,
    ) = compute_aurc_eaurc_ner_flatten_without_other(
        evalprediction,
        label_map,
    )
    aurc_flatten_span, eaurc_flatten_span = compute_aurc_eaurc_ner_flatten_span(
        evalprediction,
        label_map,
        False,
    )
    aurc_sentence, eaurc_sentence = compute_aurc_eaurc_ner_sentence(
        evalprediction,
    )
    (
        aurc_sentence_without_other,
        eaurc_sentence_without_other,
    ) = compute_aurc_eaurc_ner_sentence_without_other(
        evalprediction,
        label_map,
    )
    aurc_sentence_span, eaurc_sentence_span = compute_aurc_eaurc_ner_sentence_span(
        evalprediction,
        label_map,
        False,
    )

    metrics = {
        "accuracy_score": accuracy_score(out_label_list, preds_list),
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "expected_calibration_error_sentence": compute_calibration_error_ner_sentence(
            evalprediction,
            "ece",
        ),
        "expected_calibration_error_sentence_without_other": compute_calibration_error_ner_sentence_without_other(
            evalprediction, label_map, "ece"
        ),
        "expected_calibration_error_sentence_span": compute_calibration_error_ner_sentence_span(
            evalprediction,
            label_map,
            False,
            "ece",
        ),
        "expected_calibration_error_flatten": compute_calibration_error_ner_flatten(
            evalprediction,
            "ece",
        ),
        "expected_calibration_error_flatten_without_other": compute_calibration_error_ner_flatten_without_other(
            evalprediction, label_map, "ece"
        ),
        "expected_calibration_error_flatten_span": compute_calibration_error_ner_flatten_span(
            evalprediction,
            label_map,
            False,
            "ece",
        ),
        "maximized_calibration_error_sentence": compute_calibration_error_ner_sentence(
            evalprediction,
            "mce",
        ),
        "maximized_calibration_error_sentence_without_other": compute_calibration_error_ner_sentence_without_other(
            evalprediction, label_map, "mce"
        ),
        "maximized_calibration_error_sentence_span": compute_calibration_error_ner_sentence_span(
            evalprediction,
            label_map,
            False,
            "mce",
        ),
        "maximized_calibration_error_flatten": compute_calibration_error_ner_flatten(
            evalprediction,
            "mce",
        ),
        "maximized_calibration_error_flatten_without_other": compute_calibration_error_ner_flatten_without_other(
            evalprediction, label_map, "mce"
        ),
        "maximized_calibration_error_flatten_span": compute_calibration_error_ner_flatten_span(
            evalprediction,
            label_map,
            False,
            "mce",
        ),
        "roc_auc_sentence_macro": compute_roc_auc_ner_sentence(
            evalprediction,
            label_map,
            average="macro",
        ),
        "roc_auc_sentence_weighted": compute_roc_auc_ner_sentence(
            evalprediction, label_map, average="weighted"
        ),
        "roc_auc_sentence_without_other_macro": compute_roc_auc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "roc_auc_sentence_without_other_weighted": compute_roc_auc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "roc_auc_sentence_span": compute_roc_auc_ner_sentence_span(
            evalprediction,
            label_map,
            use_mean=False,
        ),
        "roc_auc_sentence_span_skip": compute_roc_auc_ner_sentence_span(
            evalprediction,
            label_map,
            use_mean=False,
            use_skip=True,
        ),
        "roc_auc_flatten_macro": compute_roc_auc_ner_flatten(
            evalprediction,
            average="macro",
        ),
        "roc_auc_flatten_weighted": compute_roc_auc_ner_flatten(
            evalprediction,
            average="weighted",
        ),
        "roc_auc_flatten_without_other_macro": compute_roc_auc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "roc_auc_flatten_without_other_weighted": compute_roc_auc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "roc_auc_flatten_span": compute_roc_auc_ner_flatten_span(
            evalprediction,
            label_map,
            ignore_index=None,
            use_mean=False,
        ),
        "roc_auc_flatten_span_skip": compute_roc_auc_ner_flatten_span(
            evalprediction,
            label_map,
            ignore_index=None,
            use_mean=False,
            use_skip=True,
        ),
        "auprc_sentence_macro": compute_auprc_ner_sentence(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_sentence_weighted": compute_auprc_ner_sentence(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_sentence_without_other_macro": compute_auprc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_sentence_without_other_weighted": compute_auprc_ner_sentence_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_sentence_span": compute_auprc_ner_sentence_span(
            evalprediction,
            label_map,
            use_mean=False,
        ),
        "auprc_flatten_macro": compute_auprc_ner_flatten(
            evalprediction,
            average="macro",
        ),
        "auprc_flatten_weighted": compute_auprc_ner_flatten(
            evalprediction,
            average="weighted",
        ),
        "auprc_flatten_without_other_macro": compute_auprc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="macro",
        ),
        "auprc_flatten_without_other_weighted": compute_auprc_ner_flatten_without_other(
            evalprediction,
            label_map,
            average="weighted",
        ),
        "auprc_flatten_span": compute_auprc_ner_flatten_span(
            evalprediction,
            label_map,
            ignore_index=None,
            use_mean=False,
        ),
        "aurc_sentence": aurc_sentence,
        "aurc_sentence_without_other": aurc_sentence_without_other,
        "aurc_sentence_span": aurc_sentence_span,
        "aurc_flatten": aurc_flatten,
        "aurc_flatten_without_other": aurc_flatten_without_other,
        "aurc_flatten_span": aurc_flatten_span,
        "eaurc_sentence": eaurc_sentence,
        "eaurc_sentence_without_other": eaurc_sentence_without_other,
        "eaurc_sentence_span": eaurc_sentence_span,
        "eaurc_flatten": eaurc_flatten,
        "eaurc_flatten_without_other": eaurc_flatten_without_other,
        "eaurc_flatten_span": eaurc_flatten_span,
    }

    if evalprediction.losses is not None:
        metrics.update({"loss": np.asarray(evalprediction.losses).sum()})

    # save span probabilities
    if split == "test":
        save_sentence_span_probabilities(
            evalprediction,
            os.path.join(output_path, f"{split}_span_probs_labels_{str(seed)}.pkl"),
            label_map,
        )

    return metrics


def evaluation(
    steps: Optional[int],
    model: Union[PreTrainedModel, list[PreTrainedModel]],
    dataloader: DataLoader,
    label_map: dict,
    output_path: str,
    calibration_algorithm: Optional[str] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    writer: Optional[SummaryWriter] = None,
    num_monte_carlo: int = 20,
    split: str = "test",
    seed: int = 1,
    datastore: Optional[list[dict]] = None,
) -> dict[str, float]:
    """Evaluation with dataloader

    Args:
        steps (int): now steps
        model (Union[PreTrainedModel, list[PreTrainedModel]]): model to evaluate
        dataloader (DataLoader): dataloader for evaluation
        label_map (dict): label id to label name
        calibration_algorithm (Optional[str], optional): confidence calibration algorithm. Defaults to None.
        device (_type_, optional): _description_. Defaults to torch.device( "cuda:0" if torch.cuda.is_available() else "cpu" ).
        writer (Optional[SummaryWriter], optional): _description_. Defaults to None.

    Returns:
        dict[str, float]: evaluation metrics
    """
    logger.info("=== make eval prediction ===")
    evalprediction = make_eval_prediction(
        model,
        dataloader,
        label_map,
        output_path,
        seed,
        calibration_algorithm,
        device,
        num_monte_carlo,
        datastore,
        split,
    )
    logger.info("=== computing metrics ===")
    metrics = compute_metrics(evalprediction, label_map, output_path, split, seed)

    return metrics
