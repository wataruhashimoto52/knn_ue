from typing import Optional

import numpy as np
import torch
from seqeval.metrics.sequence_labeling import get_entities
from torchmetrics.functional.classification import (
    binary_average_precision,
    multiclass_average_precision,
)

from .schemas import EvalPredictionV2


def compute_auprc_ner_flatten(
    evalprediction: EvalPredictionV2,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> float:
    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    if evalprediction.tags:  # with CRF
        final_confidences = np.asarray(
            [
                p
                for (p, a) in zip(base_confidences, evalprediction.attentions.flatten())
                if int(a) == 1
            ]
        )
    else:
        final_confidences = np.asarray(
            [p for (p, l) in zip(base_confidences, base_labels) if l != -100]
        )

    final_labels = np.asarray([l for l in base_labels if l != -100])

    score = multiclass_average_precision(
        preds=torch.from_numpy(final_confidences),
        target=torch.from_numpy(final_labels),
        num_classes=confidences.shape[-1],
        average=average,
        ignore_index=ignore_index,
    )

    return float(score)


def compute_auprc_ner_flatten_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> float:

    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    if evalprediction.tags:  # with CRF
        processed_confidences = np.asarray(
            [
                p
                for (p, a) in zip(base_confidences, evalprediction.attentions.flatten())
                if int(a) == 1
            ]
        )
    else:
        processed_confidences = np.asarray(
            [p for (p, l) in zip(base_confidences, base_labels) if l != -100]
        )

    processed_labels = np.asarray([l for l in base_labels if l != -100])

    # Oタグを指定
    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    without_o_predictions = [
        p
        for (p, l) in zip(processed_confidences, processed_labels)
        if np.argmax(np.asarray(p)) != o_tag
    ]

    without_o_labels = [
        l
        for (p, l) in zip(processed_confidences, processed_labels)
        if np.argmax(np.asarray(p)) != o_tag
    ]

    if not without_o_predictions:
        return None

    score = multiclass_average_precision(
        preds=torch.tensor(np.asarray(without_o_predictions)),
        target=torch.tensor(np.asarray(without_o_labels)),
        num_classes=confidences.shape[-1],
        average=average,
        ignore_index=ignore_index,
    )

    return float(score)


def compute_auprc_ner_flatten_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    use_mean: bool = False,
) -> float:

    confidences = evalprediction.predictions

    base_confidences = np.reshape(
        confidences, (-1, confidences.shape[-1])
    )  # (length, num_labels)
    base_labels = evalprediction.label_ids.flatten()  # (length, )

    if evalprediction.tags:  # with CRF
        processed_confidences = np.asarray(
            [
                p
                for (p, a) in zip(base_confidences, evalprediction.attentions.flatten())
                if int(a) == 1
            ]
        )
    else:
        processed_confidences = np.asarray(
            [p for (p, l) in zip(base_confidences, base_labels) if l != -100]
        )

    processed_labels = np.asarray([l for l in base_labels if l != -100])
    processed_label_entities = get_entities([label_map[l] for l in processed_labels])

    argmax_prediction = np.argmax(processed_confidences, 1)
    processed_predicted_entities = get_entities(
        [label_map[p] for p in argmax_prediction]
    )

    if not processed_predicted_entities:
        return None

    assert len(processed_confidences) == len(processed_labels)

    preds = []
    trues = []
    for i in range(len(processed_confidences)):
        e_true = [ent for ent in processed_label_entities if ent[1] == i]
        e_pred = [ent for ent in processed_predicted_entities if ent[1] == i]

        if not e_true and not e_pred:
            continue
        if e_pred:
            if use_mean:
                conf = np.mean(
                    np.max(processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            else:
                conf = np.prod(
                    np.max(processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1)
                )
            if not e_true:
                preds.append(conf)
                trues.append(0)
            elif e_true[0] == e_pred[0]:
                preds.append(conf)
                trues.append(1)
            else:
                preds.append(conf)
                trues.append(0)

        else:  # not e_pred
            if e_true:
                preds.append(0)
                trues.append(1)
            else:  # ここに来ることはない．
                preds.append(0)
                trues.append(0)

    score = binary_average_precision(
        preds=torch.from_numpy(np.asarray(preds)),
        target=torch.from_numpy(np.asarray(trues)),
        ignore_index=ignore_index,
    )

    return float(score)


def compute_auprc_ner_sentence(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> float:
    confidences = evalprediction.predictions
    labels = evalprediction.label_ids
    attentions = evalprediction.attentions

    # Oタグを指定
    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    auprcs = []
    for i in range(len(confidences)):
        confidence = confidences[i]  # (length, num_labels)
        label = labels[i]  # (length, )

        if evalprediction.tags:  # with CRF
            final_confidences = np.asarray(
                [p for (p, a) in zip(confidence, attentions[i]) if int(a) == 1]
            )
        else:
            final_confidences = np.asarray(
                [p for (p, l) in zip(confidence, label) if l != -100]
            )

        final_labels = np.asarray([l for l in label if l != -100])

        if len(set(final_labels)) == 1 and final_labels[0] == o_tag:
            continue

        score = multiclass_average_precision(
            preds=torch.from_numpy(final_confidences),
            target=torch.from_numpy(final_labels),
            num_classes=confidences.shape[-1],
            average=average,
            ignore_index=ignore_index,
        )
        if np.isnan(score):
            continue
        auprcs.append(score)

    return np.mean(auprcs)


def compute_auprc_ner_sentence_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    average: str = "macro",
) -> float:
    confidences = evalprediction.predictions
    labels = evalprediction.label_ids
    attentions = evalprediction.attentions

    # Oタグを指定
    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    auprcs = []
    for i in range(len(confidences)):
        confidence = confidences[i]  # (length, num_labels)
        label = labels[i]  # (length, )

        if evalprediction.tags:  # with CRF
            processed_confidences = np.asarray(
                [p for (p, a) in zip(confidence, attentions[i]) if int(a) == 1]
            )
        else:
            processed_confidences = np.asarray(
                [p for (p, l) in zip(confidence, label) if l != -100]
            )

        processed_labels = np.asarray([l for l in label if l != -100])

        without_o_predictions = [
            p
            for (p, l) in zip(processed_confidences, processed_labels)
            if np.argmax(np.asarray(p)) != o_tag
        ]

        without_o_labels = [
            l
            for (p, l) in zip(processed_confidences, processed_labels)
            if np.argmax(np.asarray(p)) != o_tag
        ]

        if not without_o_predictions:
            continue

        score = multiclass_average_precision(
            preds=torch.from_numpy(processed_confidences),
            target=torch.from_numpy(processed_labels),
            num_classes=confidences.shape[-1],
            average=average,
            ignore_index=ignore_index,
        )
        if np.isnan(score):
            continue
        auprcs.append(score)

    return np.mean(auprcs)


def compute_auprc_ner_sentence_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    ignore_index: Optional[int] = None,
    use_mean: bool = False,
) -> float:

    confidences = evalprediction.predictions  # (N, seq_length, confidence)
    labels = evalprediction.label_ids  # (N, seq_length, )
    attentions = evalprediction.attentions

    auprcs = []
    for i in range(len(confidences)):

        if evalprediction.tags:  # with CRF
            processed_confidences = np.asarray(
                [p for (p, a) in zip(confidences[i], attentions[i]) if int(a) == 1]
            )
        else:
            processed_confidences = np.asarray(
                [p for (p, l) in zip(confidences[i], labels[i]) if l != -100]
            )

        processed_labels = np.asarray([l for l in labels[i] if l != -100])
        processed_label_entities = get_entities(
            [label_map[l] for l in processed_labels]
        )

        argmax_prediction = np.argmax(processed_confidences, 1)
        processed_predicted_entities = get_entities(
            [label_map[p] for p in argmax_prediction]
        )

        if not processed_predicted_entities:
            continue

        assert len(processed_confidences) == len(processed_labels)

        preds = []
        trues = []
        for i in range(len(processed_confidences)):
            e_true = [ent for ent in processed_label_entities if ent[1] == i]
            e_pred = [ent for ent in processed_predicted_entities if ent[1] == i]

            if not e_true and not e_pred:
                continue
            if e_pred:
                if use_mean:
                    conf = np.mean(
                        np.max(
                            processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1
                        )
                    )
                else:
                    conf = np.prod(
                        np.max(
                            processed_confidences[e_pred[0][1] : e_pred[0][2] + 1, :], 1
                        )
                    )
                if not e_true:
                    preds.append(conf)
                    trues.append(0)
                elif e_true[0] == e_pred[0]:
                    preds.append(conf)
                    trues.append(1)
                else:
                    preds.append(conf)
                    trues.append(0)

            else:  # not e_pred
                if e_true:
                    preds.append(0)
                    trues.append(1)
                else:  # ここに来ることはない．
                    preds.append(0)
                    trues.append(0)

        score = binary_average_precision(
            preds=torch.from_numpy(np.asarray(preds)),
            target=torch.from_numpy(np.asarray(trues)),
            ignore_index=ignore_index,
        )
        if np.isnan(score):
            continue
        auprcs.append(score)

    return np.mean(auprcs)
