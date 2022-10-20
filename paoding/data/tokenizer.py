from typing import Any

from transformers import PreTrainedTokenizer

from paoding.data.dataset import Dataset


class Tokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, **kwargs):
        super().__init__(
            padding_side="right",
            pad_token="<PAD>",
            mask_token="<MASK>",
            cls_token="<CLS>",
            sep_token="<SEP>",
            bos_token="<BOS>",
            eos_token="<EOS>",
        )
        self.special_token_to_id = {
            "<PAD>": self.pad_token_id,
            "<MASK>": self.mask_token_id,
            "<CLS>": self.cls_token_id,
            "<SEP>": self.sep_token_id,
            "<BOS>": self.bos_token_id,
            "<EOS>": self.eos_token_id,
        }
        self.special_id_to_token = {v: k for k, v in self.special_token_to_id.items()}

    def prepare(self, dataset: Dataset):
        """Prepare the tokenizer for the given dataset, such as to prepare the vocab."""
        raise NotImplementedError("This is an abstract class. Do not instantiate it directly!")

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

    @property
    def bos_token_id(self) -> int:
        return 4

    @property
    def eos_token_id(self) -> int:
        return 5

    def __repr__(self) -> str:
        return self.__class__.__name__

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens=True,
        truncation=False,
        max_length=None,
        **kwargs,
    ) -> dict[str, Any]:
        kwargs = kwargs | dict(
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
