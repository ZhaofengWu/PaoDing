import gc
import logging
import math
from pathlib import Path
import sys
from types import ModuleType, FunctionType
from typing import Generic, TypeVar, Type

import torch

from paoding.argument_parser import ArgumentParser


def gpu_tensors(precision=32):
    # Adapted from https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/3
    agg = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
                if type(obj) not in agg:
                    agg[type(obj)] = 0
                agg[type(obj)] += math.prod(obj.size()) * (precision / 8) * 1e-6
        except:
            pass
    print(f"In MB: {agg}")


def getsize(obj):
    # https://stackoverflow.com/a/30316760
    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = gc.get_referents(*need_referents)
    print(size)


def add_parent_dir_to_path(file: str):
    sys.path.append(str(Path(file).parent.parent.absolute()))


WRAPPED_CLS = TypeVar("WRAPPED_CLS")


class Lazy(Generic[WRAPPED_CLS]):
    """
    Inspired by allennlp's Lazy class.

    Usage:
    ```
    class A:
        def __init__(self, a, b, c=1, d=2):
            pass

    lazy: Lazy[A] = Lazy(A, b, d=2)
    instance: A = lazy(a, c=1)
    ```
    """

    def __init__(self, wrapped_cls: Type[WRAPPED_CLS], *args, **kwargs):
        self.wrapped_cls = wrapped_cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.wrapped_cls(*args, *self.args, **kwargs, **self.kwargs)

    def add_args(self, parser: ArgumentParser):
        self.wrapped_cls.add_args(parser)

    def load_from_checkpoint(self, *args, **kwargs):
        assert (
            len(self.args) == 0
        ), "pytorch-lightning only allows class construction in load_from_checkpoint to take kwargs"
        return self.wrapped_cls.load_from_checkpoint(*args, **(kwargs | self.kwargs))


class LazyLogger:
    """
    Usually loggers are initialized in the global namespace after imports, but we usually don't
    intialize the logging module after starting to process the args, e.g. to determine the output
    directory. So the global logger needs to be a lazy one.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = None

    def init_logger(self) -> logging.Logger:
        if self.logger is None:
            self.logger = logging.getLogger(self.name)
        return self.logger

    def debug(self, *args, **kwargs):
        return self.init_logger().debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        return self.init_logger().info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return self.init_logger().warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return self.init_logger().error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        return self.init_logger().exception(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return self.init_logger().critical(*args, **kwargs)

    def fatal(self, *args, **kwargs):
        return self.init_logger().fatal(*args, **kwargs)


def get_logger(name: str) -> LazyLogger:
    return LazyLogger(name)
