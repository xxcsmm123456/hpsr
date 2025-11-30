import torch
import shutil
from math import log10
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calc_psnr(a, b, scale=None,max_value=1):
    return 10. * log10(1. / ((a - b) ** 2).mean())

def save_checkpoint(state, path, is_best):
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, path.replace('latest.pth', 'best.pth'))
