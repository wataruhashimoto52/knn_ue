import copy
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class Fields(BaseModel):
    raw_tokens: list[str]
    pieces: list[str]
    offsets: Optional[list[tuple[int, int]]]


@dataclass(init=False, repr=False)
class Token:
    """
    A simple token representation, keeping track of the token's text, offset in the passage it was
    taken from, POS tag, dependency relation, and similar information.  These fields match spacy's
    exactly, so we can just use a spacy token for this.
    # Parameters
    text : `str`, optional
        The original text represented by this token.
    idx : `int`, optional
        The character offset of this token into the tokenized passage.
    idx_end : `int`, optional
        The character offset one past the last character in the tokenized passage.
    lemma_ : `str`, optional
        The lemma of this token.
    pos_ : `str`, optional
        The coarse-grained part of speech of this token.
    tag_ : `str`, optional
        The fine-grained part of speech of this token.
    dep_ : `str`, optional
        The dependency relation for this token.
    ent_type_ : `str`, optional
        The entity type (i.e., the NER tag) for this token.
    text_id : `int`, optional
        If your tokenizer returns integers instead of strings (e.g., because you're doing byte
        encoding, or some hash-based embedding), set this with the integer.  If this is set, we
        will bypass the vocabulary when indexing this token, regardless of whether `text` is also
        set.  You can `also` set `text` with the original text, if you want, so that you can
        still use a character-level representation in addition to a hash-based word embedding.
    type_id : `int`, optional
        Token type id used by some pretrained language models like original BERT
        The other fields on `Token` follow the fields on spacy's `Token` object; this is one we
        added, similar to spacy's `lex_id`.
    """

    __slots__ = [
        "text",
        "idx",
        "idx_end",
        "lemma_",
        "pos_",
        "tag_",
        "dep_",
        "ent_type_",
        "text_id",
        "type_id",
    ]
    # Defining the `__slots__` of this class is an optimization that dramatically reduces
    # the size in memory of a `Token` instance. The downside of using `__slots__`
    # with a dataclass is that you can't assign default values at the class level,
    # which is why we need a custom `__init__` function that provides the default values.

    text: Optional[str]
    idx: Optional[int]
    idx_end: Optional[int]
    lemma_: Optional[str]
    pos_: Optional[str]
    tag_: Optional[str]
    dep_: Optional[str]
    ent_type_: Optional[str]
    text_id: Optional[int]
    type_id: Optional[int]

    def __init__(
        self,
        text: str = None,
        idx: int = None,
        idx_end: int = None,
        lemma_: str = None,
        pos_: str = None,
        tag_: str = None,
        dep_: str = None,
        ent_type_: str = None,
        text_id: int = None,
        type_id: int = None,
    ) -> None:
        assert text is None or isinstance(
            text, str
        )  # Some very hard to debug errors happen when this is not true.
        self.text = text
        self.idx = idx
        self.idx_end = idx_end
        self.lemma_ = lemma_
        self.pos_ = pos_
        self.tag_ = tag_
        self.dep_ = dep_
        self.ent_type_ = ent_type_
        self.text_id = text_id
        self.type_id = type_id

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()

    def ensure_text(self) -> str:
        """
        Return the `text` field, raising an exception if it's `None`.
        """
        if self.text is None:
            raise ValueError("Unexpected null text for token")
        else:
            return self.text


def show_token(token: Token) -> str:
    return (
        f"{token.text} "
        f"(idx: {token.idx}) "
        f"(idx_end: {token.idx_end}) "
        f"(lemma: {token.lemma_}) "
        f"(pos: {token.pos_}) "
        f"(tag: {token.tag_}) "
        f"(dep: {token.dep_}) "
        f"(ent_type: {token.ent_type_}) "
        f"(text_id: {token.text_id}) "
        f"(type_id: {token.type_id}) "
    )


class Tokenizer:
    """
    A `Tokenizer` splits strings of text into tokens.  Typically, this either splits text into
    word tokens or character tokens, and those are the two tokenizer subclasses we have implemented
    here, though you could imagine wanting to do other kinds of tokenization for structured or
    other inputs.
    See the parameters to, e.g., :class:`~.SpacyTokenizer`, or whichever tokenizer
    you want to use.
    If the base input to your model is words, you should use a :class:`~.SpacyTokenizer`, even if
    you also want to have a character-level encoder to get an additional vector for each word
    token.  Splitting word tokens into character arrays is handled separately, in the
    :class:`..token_representations.TokenRepresentation` class.
    """

    default_implementation = "spacy"

    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
        """
        Batches together tokenization of several texts, in case that is faster for particular
        tokenizers.
        By default we just do this without batching.  Override this in your tokenizer if you have a
        good way of doing batched computation.
        """
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text: str) -> List[Token]:
        """
        Actually implements splitting words into tokens.
        # Returns
        tokens : `List[Token]`
        """
        raise NotImplementedError

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        """
        Adds special tokens to tokenized text. These are tokens like [CLS] or [SEP].
        Not all tokenizers do this. The default is to just return the tokens unchanged.
        # Parameters
        tokens1 : `List[Token]`
            The list of tokens to add special tokens to.
        tokens2 : `Optional[List[Token]]`
            An optional second list of tokens. This will be concatenated with `tokens1`. Special tokens will be
            added as appropriate.
        # Returns
        tokens : `List[Token]`
            The combined list of tokens, with special tokens added.
        """
        return tokens1 + (tokens2 or [])

    def num_special_tokens_for_sequence(self) -> int:
        """
        Returns the number of special tokens added for a single sequence.
        """
        return 0

    def num_special_tokens_for_pair(self) -> int:
        """
        Returns the number of special tokens added for a pair of sequences.
        """
        return 0


class AllenNLPTokenizer:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._reverse_engineer_special_tokens("a", "b")

    def _reverse_engineer_special_tokens(
        self,
        token_a: str,
        token_b: str,
    ) -> None:
        # storing the special tokens
        self.sequence_pair_start_tokens = []
        self.sequence_pair_mid_tokens = []
        self.sequence_pair_end_tokens = []
        # storing token type ids for the sequences
        self.sequence_pair_first_token_type_id = None
        self.sequence_pair_second_token_type_id = None

        # storing the special tokens
        self.single_sequence_start_tokens = []
        self.single_sequence_end_tokens = []
        # storing token type id for the sequence
        self.single_sequence_token_type_id = None

        dummy_output = self.tokenizer.encode_plus(
            token_a,
            token_b,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        dummy_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        assert dummy_a in dummy_output["input_ids"]
        dummy_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]
        assert dummy_b in dummy_output["input_ids"]
        assert dummy_a != dummy_b

        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_a = True
                assert (
                    self.sequence_pair_first_token_type_id is None
                    or self.sequence_pair_first_token_type_id == token_type_id
                ), "multiple different token type ids found for the first sequence"
                self.sequence_pair_first_token_type_id = token_type_id
                continue

            if token_id == dummy_b:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_b = True
                assert (
                    self.sequence_pair_second_token_type_id is None
                    or self.sequence_pair_second_token_type_id == token_type_id
                ), "multiple different token type ids found for the second sequence"
                self.sequence_pair_second_token_type_id = token_type_id
                continue

            token = Token(
                self.tokenizer.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)

        assert (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=True)

        # Reverse-engineer the tokenizer for one sequence
        dummy_output = self.tokenizer.encode_plus(
            token_a,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        seen_dummy_a = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError(
                        "Cannot auto-determine the number of special tokens added."
                    )
                seen_dummy_a = True
                assert (
                    self.single_sequence_token_type_id is None
                    or self.single_sequence_token_type_id == token_type_id
                ), "multiple different token type ids found for the sequence"
                self.single_sequence_token_type_id = token_type_id
                continue

            token = Token(
                self.tokenizer.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)

        assert (
            len(self.single_sequence_start_tokens)
            + len(self.single_sequence_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=False)

    def _intra_word_tokenize(
        self, string_tokens: list[str]
    ) -> tuple[list[Token], list[Optional[tuple[int, int]]]]:
        tokens = []
        offsets = []
        for token_string in string_tokens:
            wordpieces = self.tokenizer.encode_plus(
                text=token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )

            wp_ids = wordpieces["input_ids"]

            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend(
                    Token(text=wp_text, text_id=wp_id)
                    for wp_id, wp_text in zip(
                        wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids)
                    )
                )
            else:
                offsets.append(None)

        return tokens, offsets

    def add_special_tokens(
        self, tokens1: list[Token], tokens2: Optional[list[Token]] = None
    ) -> list[Token]:
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]

        # Make sure we don't change the input parameters
        tokens2 = copy.deepcopy(tokens2)

        if tokens2 is None:
            return (
                self.single_sequence_start_tokens
                + with_new_type_id(tokens1, self.single_sequence_token_type_id)  # type: ignore
                + self.single_sequence_end_tokens
            )
        else:
            return (
                self.sequence_pair_start_tokens
                + with_new_type_id(tokens1, self.sequence_pair_first_token_type_id)  # type: ignore
                + self.sequence_pair_mid_tokens
                + with_new_type_id(tokens2, self.sequence_pair_second_token_type_id)  # type: ignore
                + self.sequence_pair_end_tokens
            )

    @staticmethod
    def _increment_offsets(
        offsets: Iterable[Optional[tuple[int, int]]], increment: int
    ) -> list[Optional[tuple[int, int]]]:
        return [
            None if offset is None else (offset[0] + increment, offset[1] + increment)
            for offset in offsets
        ]

    def intra_word_tokenize(
        self, string_tokens: list[str]
    ) -> tuple[list[Token], list[Optional[tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.

        This function inserts special tokens.
        """
        tokens, offsets = self._intra_word_tokenize(string_tokens)
        tokens = self.add_special_tokens(tokens)
        offsets = self._increment_offsets(
            offsets, len(self.single_sequence_start_tokens)
        )
        return tokens, offsets

    def retokenize(
        self, tokens: list[str], truncate: bool = True
    ) -> tuple[list[str], list[Optional[tuple[int, int]]]]:
        pieces, offsets = self.intra_word_tokenize(tokens)
        pieces = list(map(str, pieces))

        # max_lengthは要調整
        if truncate and len(pieces) > self.max_length:
            pieces = pieces[: self.max_length][:-1] + [pieces[-1]]

        return pieces, offsets

    def index_sentence(
        self,
        tokens: list[str],
    ) -> Fields:
        pieces, offsets = self.retokenize(tokens, truncate=True)
        fields = {
            "raw_tokens": tokens,  # raw tokens
            "pieces": pieces,  # subword tokenized
            "offsets": offsets,  # offsets
        }
        return Fields(**fields)
