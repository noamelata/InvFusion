import functools
import os

import PIL
import numpy as np
import torch
import torchvision


def center_crop_imagenet(image_size: int, img: PIL.Image.Image):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    pil_image = img
    while min(*pil_image.size) >= 2 * image_size:
        new_size = tuple(x // 2 for x in pil_image.size)
        assert len(new_size) == 2
        pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    new_size = tuple(round(x * scale) for x in pil_image.size)
    assert len(new_size) == 2
    pil_image = pil_image.resize(new_size, resample=PIL.Image.Resampling.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

def np_to_tensor(image):
    return torch.from_numpy(image).permute(2, 0, 1)

class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, split="train", resolution=64):
        transforms = torchvision.transforms.Compose([
            functools.partial(center_crop_imagenet, resolution),
            np_to_tensor
        ])
        split = "val" if split == "test" else split
        super().__init__(root="data/imagenet", split=split, transform=transforms)
        self.num_channels = 3

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        return image, torch.nn.functional.one_hot(torch.tensor([label]), num_classes=1000)


class ImageNet64(ImageNet):
    def __init__(self, split="train"):
        super().__init__(split=split, resolution=64)


class ImageNet256(ImageNet):
    def __init__(self, split="train"):
        super().__init__(split=split, resolution=256)


class ImageNet10K64(ImageNet64):
    def __init__(self, split="val"):
        super().__init__(split=split)
        self.imgs = [x for i, x in enumerate(self.imgs) if i % 50 < 10]
        self.samples = [x for i, x in enumerate(self.samples) if i % 50 < 10]
        self.targets = [x for i, x in enumerate(self.targets) if i % 50 < 10]


class ImageNet10K256(ImageNet256):
    def __init__(self, split="val"):
        super().__init__(split=split)
        self.imgs = [x for i, x in enumerate(self.imgs) if i % 50 < 10]
        self.samples = [x for i, x in enumerate(self.samples) if i % 50 < 10]
        self.targets = [x for i, x in enumerate(self.targets) if i % 50 < 10]


class ImageNet1K64(ImageNet64):
    def __init__(self, split="val"):
        super().__init__(split=split)
        self.imgs = [x for i, x in enumerate(self.imgs) if i % 50 < 1]
        self.samples = [x for i, x in enumerate(self.samples) if i % 50 < 1]
        self.targets = [x for i, x in enumerate(self.targets) if i % 50 < 1]

if __name__ == "__main__":
    os.chdir("..")
    dataset = ImageNet64(split="val")
    print(f"Number of images in the dataset: {len(dataset)}")
    ref_image, ref_label = dataset[0]
    print(f"Image shape: {ref_image.shape}, Label shape: {ref_label.shape}")