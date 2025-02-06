
import os
import json
import logging
import time
from typing import Optional, Union

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from torchmetrics.functional.classification import binary_average_precision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedModel
from transformers.trainer_pt_utils import nested_concat
import torchmetrics
from torchmetrics.functional.classification import multiclass_calibration_error

from .data_utils import prepare_input
from .metrics_aurcs import get_aurc_eaurc
from .metrics_calibration_errors import expected_calibration_error, maximized_calibration_error
from .schemas import EvalPredictionV2, SequenceClassifierOutputConf
from .train_utils import apply_dropout
from .ood_metrics import calc_metrics

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
    all_ood_labels = None
    all_attentions = None
    all_input_ids = None
    all_knn_distances = None
    all_knn_indices = None
    all_distance_terms = None
    all_label_terms = None
    all_neighbor_scores = None
    
    uncertainty_scores = None
    all_uncertainty_scores = None
    all_tags = []
    tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
    start = 0
    ood_labels = None
    # for Deep Ensemble
    if type(model) == list:
        batches = tqdm(dataloader)
        start = time.time()
        for step, data in enumerate(batches):
            if device:
                data = prepare_input(data, device)
                
            if "ood_labels" in data:
                ood_labels = data.pop("ood_labels")
            
            sub_all_preds = []
            sub_all_losses = []

            for sub_model in model:
                with torch.inference_mode():
                    output: SequenceClassifierOutputConf = sub_model(**data)

                sub_predictions = output.confidences
                sub_losses = output.losses

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
                else nested_concat(all_input_ids, data["input_ids"], padding_index=tokenizer.pad_token_id)
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
            all_ood_labels = (
                ood_labels
                if all_ood_labels is None
                else nested_concat(all_ood_labels, ood_labels, padding_index=-100)
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
                
            if "ood_labels" in data:
                ood_labels = nested_numpify(data.pop("ood_labels"))
                
            if "MCDropout" in model.__class__.__name__:
                # model.train()
                model.apply(apply_dropout)
                confidences = []
                loss = []
                tag = []
                for i in range(num_monte_carlo):
                    with torch.inference_mode():
                        output: SequenceClassifierOutputConf = model(**data)
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
                distance_term = output.distance_term
                label_term = output.label_term
                neighbor_scores = output.neighbor_scores
                

            else:
                with torch.inference_mode():
                    output: SequenceClassifierOutputConf = model(**data)
                
                predictions = nested_numpify(output.confidences)
                losses = output.losses
                tags = output.tags
                attentions = output.attentions
                knn_distances = output.distances
                knn_indices = output.indices
                distance_term = output.distance_term
                label_term = output.label_term
                neighbor_scores = output.neighbor_scores
                uncertainty_scores = output.uncertainty_scores


            labels = nested_numpify(data["labels"])
            
            all_input_ids = (
                data["input_ids"]
                if all_input_ids is None
                else nested_concat(all_input_ids, data["input_ids"], padding_index=tokenizer.pad_token_id)
            )
            

            if ood_labels is not None:
                all_ood_labels = (
                    ood_labels
                    if all_ood_labels is None
                    else nested_concat(all_ood_labels, ood_labels, padding_index=-100)
                )

            all_preds = (
                predictions
                if all_preds is None
                else nested_concat(all_preds, predictions, padding_index=tokenizer.pad_token_id)
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
            
            if uncertainty_scores is not None:
                all_uncertainty_scores = (
                    uncertainty_scores
                    if all_uncertainty_scores is None
                    else nested_concat(all_uncertainty_scores, uncertainty_scores, padding_index=0)
                )
            
            if knn_distances is not None:
                all_knn_distances = (
                    knn_distances
                    if all_knn_distances is None
                    else nested_concat(all_knn_distances, knn_distances, padding_index=0)
                )
            
            if knn_indices is not None:
                all_knn_indices = (
                    knn_indices
                    if all_knn_indices is None
                    else nested_concat(all_knn_indices, knn_indices, padding_index=0)
                )
                
            if distance_term is not None:
                all_distance_terms = (
                    distance_term
                    if all_distance_terms is None
                    else nested_concat(all_distance_terms, distance_term, padding_index=0)
                )
            
            if label_term is not None:
                all_label_terms = (
                    label_term
                    if all_label_terms is None
                    else nested_concat(all_label_terms, label_term, padding_index=0)
                )
                
            if neighbor_scores is not None:
                all_neighbor_scores = (
                    neighbor_scores
                    if all_neighbor_scores is None
                    else nested_concat(all_neighbor_scores, neighbor_scores, padding_index=0)
                )

            if attentions is not None:
                attentions = nested_numpify(output.attentions)
                all_attentions = (
                    attentions
                    if all_attentions is None
                    else nested_concat(all_attentions, attentions, padding_index=0)
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
    
    if split == "test":
        records = []
        for j in range(all_preds.shape[0]):
            tokens = tokenizer.convert_ids_to_tokens(
                all_input_ids[j].tolist(), skip_special_tokens=True
            )
            conf = all_preds[j].max(-1)
            pred = all_preds[j].argmax(-1)
            labs = all_labels[j]
            
            record_dict = {
                "text": " ".join(
                    [
                        comp
                        for t in tokens
                        if (comp := t.replace("▁", "")) and comp != ""
                    ]
                ),
                "confidence": float(conf),
                "prediction": int(pred),
                "label": int(labs)
            }
            
            if all_knn_distances is not None:
                record_dict.update({"knn_distances": all_knn_distances[j].tolist()})
                
            if all_knn_indices is not None:
                record_dict.update({"knn_indices": all_knn_indices[j].tolist()})
                
            if all_distance_terms is not None:
                record_dict.update({"distance_term": all_distance_terms[j].tolist()})
            
            if all_label_terms is not None:
                record_dict.update({"label_term": all_label_terms[j].tolist()})
                
            if all_neighbor_scores is not None:
                record_dict.update({"neighbor_score": all_neighbor_scores[j].tolist()})
            
            records.append(record_dict)
            
        with open(os.path.join(output_path, f"test_predictions_{str(seed)}.json"), "w") as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
    

    inference_time = str(end - start)
    if split == "test":
        with open(os.path.join(output_path, f"test_inference_time_{str(seed)}.json"), "w") as f:
            json.dump(
                {
                    "inference_time": inference_time,
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    return EvalPredictionV2(
        predictions=all_preds,
        label_ids=all_labels,
        losses=all_losses,
        tags=all_tags,
        attentions=all_attentions,
        ood_labels=all_ood_labels,
        uncertainty_scores=all_uncertainty_scores,
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
    predicted_labels = evalprediction.predictions.argmax(-1)
    predicted_confidences = evalprediction.predictions.max(-1)
    uncertainty_scores = evalprediction.uncertainty_scores
    
    true_labels = evalprediction.label_ids
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels)
    brier_score = brier_score_loss(true_labels, evalprediction.predictions[:, 1])
    ece = expected_calibration_error(evalprediction.predictions, true_labels, bins=10)
    mce = maximized_calibration_error(evalprediction.predictions, true_labels, bins=10)
    
    auroc = roc_auc_score(true_labels, evalprediction.predictions[:, 1])
    precisions, recalls, thresholds = precision_recall_curve(true_labels, evalprediction.predictions[:, 1], pos_label=1)
    auprc_sklearn = auc(recalls, precisions)
    auprc_torch = binary_average_precision(torch.from_numpy(evalprediction.predictions[:, 1]), torch.from_numpy(true_labels))
    
    # calc AURC & EAURC
    errors = (true_labels != predicted_labels).astype(int)
    
    if uncertainty_scores is not None:
        _, aurc, eaurc = get_aurc_eaurc(residuals=errors, confidence=1 - uncertainty_scores)
    else:
        _, aurc, eaurc = get_aurc_eaurc(residuals=errors, confidence=predicted_confidences)

    metrics = {
        "accuracy_score": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier_score,
        "ece": ece,
        "mce": mce,
        "auroc": auroc,
        "auprc_sklearn": auprc_sklearn,
        "auprc_torch": float(auprc_torch),
        "aurc": aurc,
        "eaurc": eaurc,
        "loss": evalprediction.losses,
    }
    
    if evalprediction.losses is not None:
        metrics.update({"loss": np.asarray(evalprediction.losses).sum()})

    return metrics


def compute_metrics_multiclass(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    output_path: str,
    split: str = "test",
    seed: int = 1,
) -> dict[str, float]:
    predicted_labels = evalprediction.predictions.argmax(-1)
    predicted_confidences: np.ndarray = evalprediction.predictions.max(-1)
    true_labels = evalprediction.label_ids
    uncertainty_scores = evalprediction.uncertainty_scores
    
    micro_precision = precision_score(true_labels, predicted_labels, average="micro")
    micro_recall = recall_score(true_labels, predicted_labels, average="micro")
    micro_f1 = f1_score(true_labels, predicted_labels, average="micro")
    macro_precision = precision_score(true_labels, predicted_labels, average="macro")
    macro_recall = recall_score(true_labels, predicted_labels, average="macro")
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    accuracy = accuracy_score(true_labels, predicted_labels)
    
    is_matched = (predicted_labels == true_labels).astype(int)
    brier_score = brier_score_loss(is_matched, predicted_confidences)
    ece = float(multiclass_calibration_error(
        torch.from_numpy(evalprediction.predictions),
        torch.from_numpy(true_labels),
        num_classes=evalprediction.predictions.shape[-1],
        n_bins=10,
        norm="l1",
    ))
    mce = float(multiclass_calibration_error(
        torch.from_numpy(evalprediction.predictions),
        torch.from_numpy(true_labels),
        num_classes=evalprediction.predictions.shape[-1],
        n_bins=10,
        norm="max",
    ))

    auroc = float(torchmetrics.functional.auroc(
        torch.from_numpy(evalprediction.predictions),
        torch.from_numpy(true_labels),
        task="multiclass",
        num_classes=evalprediction.predictions.shape[-1],
    ))
    aupr = float(torchmetrics.functional.average_precision(
        torch.from_numpy(evalprediction.predictions),
        torch.from_numpy(true_labels),
        task="multiclass",
        num_classes=evalprediction.predictions.shape[-1],
    ))
    
    errors = (true_labels != predicted_labels).astype(int)
    if uncertainty_scores is not None:
        _, aurc, eaurc = get_aurc_eaurc(residuals=errors, confidence=1 - uncertainty_scores)
    else:
        _, aurc, eaurc = get_aurc_eaurc(residuals=errors, confidence=predicted_confidences)
    
    metrics = {
        "accuracy_score": accuracy,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "brier_score": brier_score,
        "ece": ece,
        "mce": mce,
        "auroc": auroc,
        "aupr": aupr,
        "aurc": aurc,
        "eaurc": eaurc,
    }
    
    if evalprediction.losses is not None:
        metrics.update({"loss": np.asarray(evalprediction.losses).sum()})
    
    return metrics


def compute_metrics_ood_detection(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    output_path: str,
    split: str = "test",
    seed: int = 1,
) -> None:
    predicted_confidences: np.ndarray = evalprediction.predictions.max(-1)
    if evalprediction.uncertainty_scores is not None:
        ood_scores = evalprediction.uncertainty_scores
    else:
        ood_scores = 1 - predicted_confidences
    ood_labels = evalprediction.ood_labels
    
    metrics = calc_metrics(ood_scores, ood_labels, pos_label=1)
    
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
    is_ood_detection: bool = False,
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
    if is_ood_detection:
        return compute_metrics_ood_detection(evalprediction, label_map, output_path, split, seed)
    
    if len(label_map) > 2:
        metrics = compute_metrics_multiclass(evalprediction, label_map, output_path, split, seed)
    else:
        metrics = compute_metrics(evalprediction, label_map, output_path, split, seed)

    return metrics
