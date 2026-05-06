import glob
import random
import os
import numpy as np
import tifffile
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch





class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = transforms.Compose(transforms_1)
        self.transform2 = transforms.Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level
        
    def __getitem__(self, index):
        with tifffile.TiffReader(self.files_A[index % len(self.files_A)]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index % len(self.files_A)]) as tif:
            img_b = tif.pages[0].asarray()

        if self.noise_level == 0:
            # if noise =0, A and B make same transform
            seed = np.random.randint(2147483647) # make a seed with numpy generator 
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_A = self.transform2(img=img_a)

            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            item_B = self.transform2(img=img_b)
        else:
            # if noise !=0, A and B make different transform
            item_A = self.transform1(img=img_a)
            item_B = self.transform1(img=img_b)
            
            
            
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ValDataset(Dataset):
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        
    def __getitem__(self, index):
        with tifffile.TiffReader(self.files_A[index % len(self.files_A)]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index % len(self.files_A)]) as tif:
            img_b = tif.pages[0].asarray()

        item_A = self.transform(img=img_a)
        if self.unaligned:
            raise NotImplemented
        else:
            item_B = self.transform(img=img_b)
        return {'A': item_A, 'B': item_B}
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
