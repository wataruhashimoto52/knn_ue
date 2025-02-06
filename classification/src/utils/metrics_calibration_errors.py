
import pickle
from typing import Union

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities
from .schemas import EvalPredictionV2, SequenceClassifierOutputConf


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Computes the top 1 accuracy of the predicted class probabilities in percent.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
    Returns:
        The top 1 accuracy in percent.
    """
    return 100.0 * np.mean(np.argmax(probabilities, axis=1) == labels)


def confidence(probabilities: np.ndarray, mean: bool = True):
    """The confidence of a prediction is the maximum of the predicted class probabilities.
    Args:
        probabilities: The predicted class probabilities.
        mean: If True, returns the average confidence over all provided predictions.
    Returns:
        The confidence.
    """
    if mean:
        return np.mean(np.max(probabilities, axis=1))
    return np.max(probabilities, axis=1)


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.
    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """

    conf = confidence(probabilities, mean=False)
    # conf = probabilities
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece


def maximized_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.
    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """

    conf = confidence(probabilities, mean=False)
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    eces = []
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = accuracy(probabilities[mask], labels[mask]) / 100
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            eces.append(np.abs(ace))

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)

    return np.max(eces)


def expected_calibration_error_for_span(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.
    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """

    conf = probabilities
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ece = 0
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = np.mean(labels[mask])
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ece += mask.mean() * np.abs(ace)

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
    return ece


def maximized_calibration_error_for_span(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> float:
    r"""Computes the Expected Calibration Error (ECE) of the predicted class probabilities.
    With accuracy `acc` and confidence `conf`, it is defined as
    :math:`ECE=\sum_{m=1}^M\frac{\left|B_m\right|}{n}\left|\mathrm{acc}(B_M)-\mathrm{conf}(B_m)\right|`
    where `n` is the number of samples and `B_m` are all samples in bin `m` from `M` equally-spaced bins.
    Source: `On Calibration of Modern Neural Networks <https://arxiv.org/pdf/1706.04599.pdf)?>`_.
    Args:
        probabilities: The predicted class probabilities.
        labels: The ground truth labels.
        bins: The number of bins into which the probabilities are discretized.
    Returns:
        The ECE alongside the average confidence, accuracy and proportion of data points in each bin respectively.
    """

    conf = probabilities
    edges = np.linspace(0, 1, bins + 1)
    bin_ace = list()
    bin_accuracy = list()
    bin_confidence = list()
    ces = []
    for i in range(bins):
        mask = np.logical_and(conf > edges[i], conf <= edges[i + 1])
        if any(mask):
            bin_acc = np.mean(labels[mask])
            bin_conf = conf[mask].mean()
            ace = bin_conf - bin_acc
            ces.append(np.abs(ace))

            bin_ace.append(ace)
            bin_accuracy.append(bin_acc)
            bin_confidence.append(bin_conf)
        else:
            bin_ace.append(0)
            bin_accuracy.append(0)
            bin_confidence.append(0)
            ces.append(0)

    return np.max(ces)


def calibration_error_without_other(
    processed_predictions: list[Union[np.ndarray, float]],
    processed_labels: list[float],
    label_map: dict,
    calibration_error_type: str = "ece",  # or mce
):
    try:
        o_tag = [k for k, v in label_map.items() if v == "O"][0]
    except IndexError:
        raise Exception

    without_o_predictions = [
        p
        for (p, l) in zip(processed_predictions, processed_labels)
        if np.argmax(np.asarray(p)) != o_tag
    ]

    without_o_labels = [
        l
        for (p, l) in zip(processed_predictions, processed_labels)
        if np.argmax(np.asarray(p)) != o_tag
    ]

    if not without_o_predictions:
        return None

    if calibration_error_type == "ece":
        ce = expected_calibration_error(
            np.asarray(without_o_predictions), np.asarray(without_o_labels), bins=10
        )
    elif calibration_error_type == "mce":
        ce = maximized_calibration_error(
            np.asarray(without_o_predictions), np.asarray(without_o_labels), bins=10
        )
    else:
        raise NotImplementedError

    return ce


def compute_calibration_error_ner_sentence(
    evalprediction: EvalPredictionV2, calibration_error_type: str = "ece"  # or mce
):
    predictions = evalprediction.predictions
    label_ids = evalprediction.label_ids

    ces = []
    for i in range(len(predictions)):
        if evalprediction.tags:  # with CRF
            pred = [
                p
                for (p, a) in zip(predictions[i], evalprediction.attentions[i])
                if int(a) == 1
            ]
        else:
            pred = [p for (p, l) in zip(predictions[i], label_ids[i]) if l != -100]

        label = [l for (p, l) in zip(predictions[i], label_ids[i]) if l != -100]

        pred = np.asarray(pred)
        label = np.asarray(label)

        if calibration_error_type == "ece":
            ce = expected_calibration_error(pred, label, bins=10)
        elif calibration_error_type == "mce":
            ce = maximized_calibration_error(pred, label, bins=10)
        else:
            raise NotImplementedError

        ces.append(ce)

    return np.mean(ces)


def compute_calibration_error_ner_sentence_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    use_mean: bool = False,
    calibration_error_type: str = "ece",  # or mce
):
    """_summary_

    Args:
        evalprediction (EvalPredictionV2): _description_
        label_map (dict): _description_
        use_mean (bool, optional): _description_. Defaults to False.
    """
    predictions = evalprediction.predictions
    label_ids = evalprediction.label_ids

    ces = []
    for i in range(len(predictions)):
        if evalprediction.tags:  # with CRF
            processed_predictions = [
                p
                for (p, a) in zip(predictions[i], evalprediction.attentions[i])
                if int(a) == 1
            ]
        else:
            processed_predictions = [
                p for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
            ]

        processed_labels = [
            l for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
        ]

        processed_label_entities = get_entities(
            [label_map[l] for l in processed_labels]
        )

        argmax_prediction = np.argmax(np.asarray(processed_predictions), 1)
        processed_predicted_entities = get_entities(
            [label_map[p] for p in argmax_prediction]
        )
        processed_label_entities_set = set(processed_label_entities)

        processed_labels_span = [
            1 if predicted_entity in processed_label_entities_set else 0
            for predicted_entity in processed_predicted_entities
        ]
        if use_mean:
            processed_predictions_span = [
                np.mean(
                    np.max(
                        np.asarray(processed_predictions)[
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
                        np.asarray(processed_predictions)[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                        1,
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]

        if not processed_predicted_entities:
            continue

        if calibration_error_type == "ece":
            ce = expected_calibration_error_for_span(
                np.asarray(processed_predictions_span),
                np.asarray(processed_labels_span),
                bins=10,
            )
        elif calibration_error_type == "mce":
            ce = maximized_calibration_error_for_span(
                np.asarray(processed_predictions_span),
                np.asarray(processed_labels_span),
                bins=10,
            )
        else:
            raise NotImplementedError

        ces.append(ce)

    return np.mean(ces)


def compute_calibration_error_ner_flatten(
    evalprediction: EvalPredictionV2, calibration_error_type: str = "ece"
):
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

    if evalprediction.tags:  # with CRF
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
    if calibration_error_type == "ece":
        ce = expected_calibration_error(
            np.asarray(processed_predictions), np.asarray(processed_labels), bins=10
        )
    elif calibration_error_type == "mce":
        ce = maximized_calibration_error(
            np.asarray(processed_predictions), np.asarray(processed_labels), bins=10
        )
    else:
        raise NotImplementedError

    return ce


def compute_calibration_error_ner_flatten_span(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    use_mean: bool = False,
    calibration_error_type: str = "ece",
):
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

    if evalprediction.tags:  # with CRF
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

    processed_label_entities = get_entities([label_map[l] for l in processed_labels])
    argmax_prediction = np.argmax(np.asarray(processed_predictions), 1)
    processed_predicted_entities = get_entities(
        [label_map[p] for p in argmax_prediction]
    )
    processed_label_entities_set = set(processed_label_entities)

    processed_labels_span = [
        1 if predicted_entity in processed_label_entities_set else 0
        for predicted_entity in processed_predicted_entities
    ]
    if use_mean:
        processed_predictions_span = [
            np.mean(
                np.max(
                    np.asarray(processed_predictions)[
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
                    np.asarray(processed_predictions)[
                        predicted_entity[1] : predicted_entity[2] + 1, :
                    ],
                    1,
                )
            )
            for predicted_entity in processed_predicted_entities
        ]

    if not processed_predicted_entities:
        return None

    if calibration_error_type == "ece":
        ce = expected_calibration_error_for_span(
            np.asarray(processed_predictions_span),
            np.asarray(processed_labels_span),
            bins=10,
        )
    elif calibration_error_type == "mce":
        ce = maximized_calibration_error_for_span(
            np.asarray(processed_predictions_span),
            np.asarray(processed_labels_span),
            bins=10,
        )
    else:
        raise NotImplementedError

    return ce


def compute_calibration_error_ner_flatten_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    calibration_error_type: str = "ece",  # or mce
):
    """(batch_size, length, num_labels) -> (batch_size * length, num_labels) とフラットにしてから，
    Oタグのみを除いた領域のECEを測る

    Args:
        evalprediction (EvalPredictionV2): _description_
        label_map (dict): _description_
    """
    predictions = np.reshape(
        evalprediction.predictions, (-1, evalprediction.predictions.shape[-1])
    )
    label_ids = evalprediction.label_ids.flatten()

    if evalprediction.tags:  # with CRF
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

    ce = calibration_error_without_other(
        processed_predictions=processed_predictions,
        processed_labels=processed_labels,
        label_map=label_map,
        calibration_error_type=calibration_error_type,
    )
    return ce


def compute_calibration_error_ner_sentence_without_other(
    evalprediction: EvalPredictionV2,
    label_map: dict,
    calibration_error_type: str = "ece",
):
    predictions = evalprediction.predictions
    label_ids = evalprediction.label_ids

    ces = []
    for i in range(len(predictions)):
        if evalprediction.tags:  # with CRF
            processed_predictions = [
                p
                for (p, a) in zip(predictions[i], evalprediction.attentions[i])
                if int(a) == 1
            ]
        else:
            processed_predictions = [
                p for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
            ]

        processed_labels = [
            l for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
        ]

        ce = calibration_error_without_other(
            processed_predictions=processed_predictions,
            processed_labels=processed_labels,
            label_map=label_map,
            calibration_error_type=calibration_error_type,
        )
        if ce is None:
            continue
        ces.append(ce)

    return np.mean(ces)


def save_sentence_span_probabilities(
    evalprediction: EvalPredictionV2,
    output_path: str,
    label_map: dict,
    use_mean: bool = False,
) -> None:
    predictions = evalprediction.predictions
    label_ids = evalprediction.label_ids

    span_probabilities = []
    span_labels = []
    for i in range(len(predictions)):
        if evalprediction.tags:  # with CRF
            processed_predictions = [
                p
                for (p, a) in zip(predictions[i], evalprediction.attentions[i])
                if int(a) == 1
            ]
        else:
            processed_predictions = [
                p for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
            ]

        processed_labels = [
            l for (p, l) in zip(predictions[i], label_ids[i]) if l != -100
        ]

        processed_label_entities = get_entities(
            [label_map[l] for l in processed_labels]
        )

        argmax_prediction = np.argmax(np.asarray(processed_predictions), 1)
        processed_predicted_entities = get_entities(
            [label_map[p] for p in argmax_prediction]
        )
        processed_label_entities_set = set(processed_label_entities)

        processed_labels_span = [
            1 if predicted_entity in processed_label_entities_set else 0
            for predicted_entity in processed_predicted_entities
        ]
        if use_mean:
            processed_predictions_span = [
                np.mean(
                    np.max(
                        np.asarray(processed_predictions)[
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
                        np.asarray(processed_predictions)[
                            predicted_entity[1] : predicted_entity[2] + 1, :
                        ],
                        1,
                    )
                )
                for predicted_entity in processed_predicted_entities
            ]

        if not processed_predicted_entities:
            continue

        span_probabilities.append(np.asarray(processed_predictions_span))
        span_labels.append(np.asarray(processed_labels_span))

    output_dict = {
        "span_probabilities": span_probabilities,
        "span_labels": span_labels,
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
