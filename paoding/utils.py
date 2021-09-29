import gc
import logging
import math
from pathlib import Path
import sys

import torch


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


def add_parent_dir_to_path(file: str):
    sys.path.append(str(Path(file).parent.parent.absolute()))


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
