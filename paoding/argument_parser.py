import argparse
from collections.abc import KeysView


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self._arguments_set = set()
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs) -> None:
        """
        Add a conflict resolution mechanism.

        NOTE: Conflicts are only resolved when args/kwargs completely match. If the flag name
        matches but not other attributes, the super class's add_argument will still raise an error.

        NOTE: This method in the superclass returns the created action, but I don't know how to
        easily do that in the case of a "cache hit", so I'm disabling return altogether. We don't
        typically need it anyway, and when we do I can figure something out.
        """
        # manually patch a few common non-hashable cases
        def make_hashable(e):
            if isinstance(e, (list, KeysView, set)):
                return tuple(e)
            else:
                return e

        hash_ = tuple(
            [make_hashable(a) for a in args] + [(k, make_hashable(v)) for k, v in kwargs.items()]
        )
        if hash_ in self._arguments_set:
            return
        self._arguments_set.add(hash_)
        super().add_argument(*args, **kwargs)
