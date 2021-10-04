from typing import Any, Union

from transformers import PreTrainedTokenizerBase


class Tokenizer(PreTrainedTokenizerBase):
    def __init__(self, **kwargs):
        super().__init__(padding_side="right", pad_token="<PAD>")

    @property
    def pad_token_id(self) -> int:
        return 0

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __call__(
        self,
        text: Union[str, list[str]],
        add_special_tokens=False,
        truncation=False,
        max_length=None,
    ) -> dict[str, Any]:
        kwargs = dict(
            add_special_tokens=add_special_tokens, truncation=truncation, max_length=max_length
        )
        if isinstance(text, list):
            if len(text) == 0:
                return []
            result = self._tokenize(text[0], **kwargs)
            results = {k: [v] for k, v in result.items()}
            for t in text[1:]:
                result = self._tokenize(t, **kwargs)
                assert results.keys() == result.keys()
                for k, v in result.items():
                    results[k].append(v)
            return results
        else:
            return self._tokenize(text, **kwargs)

    def _tokenize(
        self, text: str, add_special_tokens=False, truncation=False, max_length=None
    ) -> dict[str, Any]:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    # The following methods assume 1-1 mapping between tokens and IDs. If this doesn't apply,
    # override.

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_token_to_id(self, token: str) -> int:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.convert_id_to_token(id) for id in ids]

    def convert_id_to_token(self, id: int) -> str:
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    # Below are methods implemented in the superclass. We don't want the superclass implementation
    # do be accidentally used.

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    @classmethod
    def save_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def tokenize(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def encode(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def encode_plus(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def batch_encode_plus(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def prepare_for_model(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def batch_decode(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def decode(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

    def prepare_seq2seq_batch(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")
