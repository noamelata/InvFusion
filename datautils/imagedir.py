from glob import glob

import numpy as np
import torch
from PIL import Image


def pil_to_tensor(image):
    return torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)


class ImageDir(torch.utils.data.Dataset):
    def __init__(self, split="train", imsize=64):
        super().__init__()
        self.images = glob(f"data/imagedir/*.png")
        self.num_channels = 3

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return pil_to_tensor(Image.open(self.images[index])), torch.zeros((0,))

