from typing import List

import numpy as np
from pydantic import BaseModel


class TokensMapping(BaseModel):
    before_tokens: List[str]
    before_type: List[str]
    after_tokens: List[str]
    after_type: List[str]
    soft_label_ratio: float


labels = ["B-ORG", "I-ORG", "B-PER", "I-PER", "O"]

id2label = {0: "B-ORG", 1: "I-ORG", 2: "B-PER", 3: "I-PER", 4: "O"}

label2id = {"B-ORG": 0, "I-ORG": 1, "B-PER": 2, "I-PER": 3, "O": 4}

num_labels = len(labels)

input_tokens = ["hoge", "fuga", "is", "ea", "ab", "cd", "d"]
input_tags_string = ["B-ORG", "I-ORG", "O", "O", "B-PER", "I-PER", "I-PER"]
input_tags = [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0],
]

sample_tokens_mapping = [
    TokensMapping(
        before_tokens=["hoge", "fuga"],
        before_type=["B-ORG", "I-ORG"],
        after_tokens=["piyo", "bara", "hina"],
        after_type=["B-PER", "I-PER", "I-PER"],
        soft_label_ratio=0.7,
    ),
    TokensMapping(
        before_tokens=["ab", "cd", "d"],
        before_type=["B-PER", "I-PER", "I-PER"],
        after_tokens=["ef", "gh"],
        after_type=["B-ORG", "I-ORG"],
        soft_label_ratio=0.1,
    ),
]


def replace_tokens(
    tokens: list[str], tokens_mappings: list[TokensMapping]
) -> list[str]:
    output_tokens: list[str] = []
    i = 0

    while i < len(tokens):
        matched = False

        for mapping in tokens_mappings:
            before_tokens = mapping.before_tokens
            after_tokens = mapping.after_tokens

            if tokens[i : i + len(before_tokens)] == before_tokens:
                output_tokens.extend(after_tokens)
                i += len(before_tokens)
                matched = True
                break

        if not matched:
            output_tokens.append(tokens[i])
            i += 1

    return output_tokens


def generate_soft_labels(
    before_type: list[str],
    after_type: list[str],
    after_tokens_length: int,
    label2id: dict[str, int],
    num_labels: int,
    hyper_lambda: float,
) -> list[list[float]]:
    before_entity_type = before_type[0][2:]
    after_entity_type = after_type[0][2:]

    output_tags = []
    for i in range(after_tokens_length):
        base_identity = np.identity(num_labels)
        if i == 0:
            before_array = (
                base_identity[label2id["B-" + before_entity_type]] * hyper_lambda
            )
            after_array = base_identity[label2id["B-" + after_entity_type]] * (
                1 - hyper_lambda
            )
        else:
            before_array = (
                base_identity[label2id["I-" + before_entity_type]] * hyper_lambda
            )
            after_array = base_identity[label2id["I-" + after_entity_type]] * (
                1 - hyper_lambda
            )

        arr: np.ndarray = before_array + after_array
        arr = arr.tolist()

        output_tags.append(arr)

    return output_tags


def replace_tokens_and_generate_soft_labels(
    tokens: list[str], input_tags: list[list[int]], tokens_mappings: list[TokensMapping]
) -> tuple[list[str], list[list[float]]]:
    output_tokens: list[str] = []
    output_tags: list[list[float]] = []
    i = 0

    while i < len(tokens):
        matched = False

        for mapping in tokens_mappings:
            before_tokens = mapping.before_tokens
            after_tokens = mapping.after_tokens

            before_tags = mapping.before_type
            after_tags = mapping.after_type

            if tokens[i : i + len(before_tokens)] == before_tokens:
                output_tokens.extend(after_tokens)
                output_tags.extend(
                    generate_soft_labels(
                        before_type=before_tags,
                        after_type=after_tags,
                        label2id=label2id,
                        num_labels=num_labels,
                        after_tokens_length=len(after_tokens),
                        hyper_lambda=mapping.soft_label_ratio,
                    )
                )
                i += len(before_tokens)
                matched = True
                break

        if not matched:
            output_tokens.append(tokens[i])
            output_tags.append(input_tags[i])
            i += 1

    return output_tokens, output_tags


# output_tokens = ['piyo', 'bara', 'hina', 'is', 'ea', 'ef', 'gh']
output_tokens, output_tags = replace_tokens_and_generate_soft_labels(
    tokens=input_tokens,
    input_tags=input_tags,
    tokens_mappings=sample_tokens_mapping,
)

print(output_tokens)
print(output_tags)
