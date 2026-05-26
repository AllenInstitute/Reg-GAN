import glob
import numpy as np
import random
from pathlib import Path
import tifffile
from skimage import io
from torch.utils.data import Dataset
from albumentations import Compose
import torch


def _load_mask(masks_path: str, image_stem: str) -> np.ndarray:
    mask_path = Path(masks_path) / f"{image_stem}_mask.png"

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: expected {mask_path}")

    mask = io.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]

    return mask.astype('uint8')

class ImageDataset(Dataset):
    def __init__(self, root, noise_level=None, count=None, transforms_1=None, transforms_2=None, unaligned=False, *, masks_path):
        self.transform1 = Compose(transforms_1)
        self.transform2 = Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level = noise_level
        self.masks_path = masks_path

    def __getitem__(self, index):
        index_A = index % len(self.files_A)
        index_B = random.randint(0, len(self.files_B) - 1) if self.unaligned else index % len(self.files_B)

        with tifffile.TiffReader(self.files_A[index_A]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index_B]) as tif:
            img_b = tif.pages[0].asarray()

        mask_a = _load_mask(self.masks_path, Path(self.files_A[index_A]).stem)
        mask_b = _load_mask(self.masks_path, Path(self.files_B[index_B]).stem)

        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            transformed_a = self.transform2(image=img_a, mask=mask_a)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            transformed_b = self.transform2(image=img_b, mask=mask_b)
        else:
            # if noise !=0, A and B make different transform
            transformed_a = self.transform1(image=img_a, mask=mask_a)
            transformed_b = self.transform1(image=img_b, mask=mask_b)

        img_a = transformed_a['image']
        mask_a = transformed_a['mask']

        img_b = transformed_b['image']
        mask_b = transformed_b['mask']

        return {
            'A': img_a,
            'B': img_b,
            'A_mask': mask_a.float().unsqueeze(0),
            'B_mask': mask_b.float().unsqueeze(0),
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False, *, masks_path):
        self.transform = Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.masks_path = masks_path

    def __getitem__(self, index):
        index_A = index % len(self.files_A)
        index_B = random.randint(0, len(self.files_B) - 1) if self.unaligned else index % len(self.files_B)

        with tifffile.TiffReader(self.files_A[index_A]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index_B]) as tif:
            img_b = tif.pages[0].asarray()

        mask_a = _load_mask(self.masks_path, Path(self.files_A[index_A]).stem)
        mask_b = _load_mask(self.masks_path, Path(self.files_B[index_B]).stem)

        item_A = self.transform(image=img_a, mask=mask_a)
        if self.unaligned:
            raise NotImplemented
        else:
            item_B = self.transform(image=img_b, mask=mask_b)
        return {
            'A': item_A['image'],
            'B': item_B['image'],
            'A_mask': item_A['mask'].float().unsqueeze(0),
            'B_mask': item_B['mask'].float().unsqueeze(0),
        }
    def __len__(self):
        if self.unaligned:
            return max(len(self.files_A), len(self.files_B))
        return min(len(self.files_A), len(self.files_B))
