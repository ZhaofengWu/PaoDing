import math
import gc

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
