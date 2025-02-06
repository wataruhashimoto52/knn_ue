import itertools
from typing import Optional

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

from .schemas import EvalPredictionV2


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


def compute_aurc_eaurc_ner_flatten(
    evalprediction: EvalPredictionV2,
) -> tuple[float, float]:

    if evalprediction.tags:  # with CRF
        predictions = evalprediction.tags
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


def compute_aurc_eaurc_ner_flatten_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
) -> tuple[Optional[float], Optional[float]]:
    confidences = np.reshape(
        evalprediction.predictions, (-1, len(label_map))
    )  # (N, num_labels)

    base_labels = evalprediction.label_ids.flatten()  # (N, )

    # use tag from forward-backward algorithm's output confidence (CRF)
    # use softmax confidence (not-CRF)
    if evalprediction.tags:
        base_confidences = np.asarray(
            [
                p
                for (p, a) in zip(confidences, evalprediction.attentions.flatten())
                if int(a) == 1
            ]
        )
    else:
        base_confidences = np.asarray(
            [c for (c, l) in zip(confidences, base_labels) if l != -100]
        )

    base_labels = np.asarray([l for l in base_labels if l != -100])

    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    without_o_confidences = np.asarray(
        [p for p in base_confidences if np.argmax(np.asarray(p)) != o_tag]
    )  # (N, num_labels)
    without_o_labels = np.asarray(
        [
            l
            for (p, l) in zip(base_confidences, base_labels)
            if np.argmax(np.asarray(p)) != o_tag
        ]
    )  # (N, )

    try:
        without_o_predictions = np.argmax(without_o_confidences, 1)
        errors = (without_o_labels != without_o_predictions).astype(int)
    except np.AxisError:
        return None, None

    if not without_o_confidences.tolist():
        return None, None

    _, aurc, eaurc = get_aurc_eaurc(
        residuals=errors,
        confidence=np.asarray([np.max(c) for c in without_o_confidences]),
    )

    return aurc, eaurc


def compute_aurc_eaurc_ner_flatten_span(
    evalprediction: EvalPredictionV2, label_map: dict, use_mean: bool = False
) -> tuple[float, float]:

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


def compute_aurc_eaurc_ner_sentence(
    evalprediction: EvalPredictionV2,
):
    base_confidences = evalprediction.predictions
    base_labels = evalprediction.label_ids
    base_attentions = evalprediction.attentions

    aurcs = []
    eaurcs = []
    for i in range(len(base_confidences)):
        if evalprediction.tags:  # with CRF
            processed_confidences = np.asarray(
                [
                    p
                    for (p, a) in zip(base_confidences[i], base_attentions[i])
                    if int(a) == 1
                ]
            )
        else:
            processed_confidences = np.asarray(
                [p for (p, l) in zip(base_confidences[i], base_labels[i]) if l != -100]
            )

        processed_labels = np.asarray([l for l in base_labels[i] if l != -100])

        processed_predictions = np.argmax(processed_confidences, 1)

        errors = (processed_labels != processed_predictions).astype(int)

        curve, aurc, eaurc = get_aurc_eaurc(
            residuals=errors,
            confidence=np.asarray([np.max(c) for c in processed_confidences]),
        )
        if np.isnan(aurc) or np.isnan(eaurc):
            continue
        aurcs.append(aurc)
        eaurcs.append(eaurc)

    return np.mean(aurcs), np.mean(eaurcs)


def compute_aurc_eaurc_ner_sentence_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
) -> tuple[float, float]:
    base_confidences = evalprediction.predictions
    base_labels = evalprediction.label_ids
    base_attentions = evalprediction.attentions

    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    aurcs = []
    eaurcs = []
    for i in range(len(base_confidences)):
        if evalprediction.tags:  # with CRF
            processed_confidences = [
                p
                for (p, a) in zip(base_confidences[i], base_attentions[i])
                if int(a) == 1
            ]
        else:
            processed_confidences = [
                p for (p, l) in zip(base_confidences[i], base_labels[i]) if l != -100
            ]

        processed_labels = [l for l in base_labels[i] if l != -100]

        final_confidences = [
            c for c in processed_confidences if np.argmax(np.asarray(c)) != o_tag
        ]
        final_labels = [
            l
            for (c, l) in zip(processed_confidences, processed_labels)
            if np.argmax(np.asarray(c)) != o_tag
        ]

        if not final_confidences:
            continue

        final_confidences = np.asarray(final_confidences)
        final_labels = np.asarray(final_labels)
        final_predictions = np.argmax(final_confidences, 1)

        errors = (final_labels != final_predictions).astype(int)
        curve, aurc, eaurc = get_aurc_eaurc(
            residuals=errors,
            confidence=np.asarray([np.max(c) for c in final_confidences]),
        )
        if np.isnan(aurc) or np.isnan(eaurc):
            continue

        aurcs.append(aurc)
        eaurcs.append(eaurc)

    return np.mean(aurcs), np.mean(eaurcs)


def compute_aurc_eaurc_ner_sentence_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    use_mean: bool = False,
) -> tuple[float, float]:
    base_confidences = evalprediction.predictions
    base_labels = evalprediction.label_ids
    base_attentions = evalprediction.attentions

    aurcs = []
    eaurcs = []
    for i in range(len(base_confidences)):
        if evalprediction.tags:  # with CRF
            processed_confidences = np.asarray(
                [
                    p
                    for (p, a) in zip(base_confidences[i], base_attentions[i])
                    if int(a) == 1
                ]
            )
        else:
            processed_confidences = np.asarray(
                [p for (p, l) in zip(base_confidences[i], base_labels[i]) if l != -100]
            )

        processed_labels = np.asarray([l for l in base_labels[i] if l != -100])
        processed_predictions = np.argmax(processed_confidences, 1)

        processed_label_entities = get_entities(
            [label_map[l] for l in processed_labels]
        )
        processed_predicted_entities = get_entities(
            [label_map[p] for p in processed_predictions]
        )

        if not processed_predicted_entities:
            continue

        processed_label_entities_set = set(processed_label_entities)

        processed_errors_span = [
            0 if predicted_entity in processed_label_entities_set else 1
            for predicted_entity in processed_predicted_entities
        ]
        if use_mean:
            processed_predictions_span = [
                np.mean(
                    np.max(
                        processed_confidences[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]
        else:
            processed_predictions_span = [
                np.prod(
                    np.max(
                        processed_confidences[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]

        curve, aurc, eaurc = get_aurc_eaurc(
            residuals=np.asarray(processed_errors_span),
            confidence=np.asarray(processed_predictions_span),
        )
        if np.isnan(aurc) or np.isnan(eaurc):
            continue
        aurcs.append(aurc)
        eaurcs.append(eaurc)

    return np.mean(aurcs), np.mean(eaurcs)
