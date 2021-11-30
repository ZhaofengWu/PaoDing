from typing import Any, Union

from transformers import PreTrainedTokenizer


class Tokenizer(PreTrainedTokenizer):
    def __init__(self, **kwargs):
        super().__init__(padding_side="right", pad_token="<PAD>", mask_token="<MASK>", cls_token="<CLS>", sep_token="<SEP>")

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def mask_token_id(self) -> int:
        return 1

    @property
    def cls_token_id(self) -> int:
        return 2

    @property
    def sep_token_id(self) -> int:
        return 3

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

    # Below are methods implemented in the superclass. We don't want the superclass implementation
    # to be accidentally used.

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

    def prepare_seq2seq_batch(self, *args, **kwargs):
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")
