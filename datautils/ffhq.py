from glob import glob

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)

class FFHQ(torch.utils.data.Dataset):
    def __init__(self, split="train", imsize=64):
        super().__init__()
        self.imsize = imsize
        self.images = glob(f"data/FFHQ/*.png")
        self.images = self.images[:60_000] if split == "train" else self.images[60_000:]
        self.num_channels = 3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return pil_to_tensor(Image.open(self.images[index]).resize((self.imsize, self.imsize))), torch.zeros((0,))
    
class FFHQ64(FFHQ):
    def __init__(self, split="train"):
        super().__init__(split=split, imsize=64)

class FFHQ256(FFHQ):
    def __init__(self, split="train"):
        super().__init__(split=split, imsize=256)
