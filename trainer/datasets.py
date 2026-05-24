import glob
import numpy as np
import tifffile
from torch.utils.data import Dataset
from albumentations import Compose
import torch
from transformers import SamModel, SamProcessor


class ImageDataset(Dataset):
    def __init__(self, root,noise_level,count = None,transforms_1=None,transforms_2=None, unaligned=False):
        self.transform1 = Compose(transforms_1)
        self.transform2 = Compose(transforms_2)
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))
        self.unaligned = unaligned
        self.noise_level =noise_level

        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self._sam_model = model
        self._sam_processor = processor

    def _get_mask(self, image: np.ndarray):
        raw_image_3channel = np.stack([image, image, image], axis=-1)
        input_boxes = [[[0, 0, image.shape[1], image.shape[0]]]]
        inputs = self._sam_processor(raw_image_3channel, input_boxes=[input_boxes], return_tensors="pt")
        with torch.no_grad():
            outputs = self._sam_model(**inputs, multimask_output=False)
        masks = self._sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                             inputs["original_sizes"].cpu(),
                                                             inputs["reshaped_input_sizes"].cpu())
        mask = masks[0][0, 0].numpy().astype('uint8')
        return mask

    def __getitem__(self, index):
        with tifffile.TiffReader(self.files_A[index % len(self.files_A)]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index % len(self.files_A)]) as tif:
            img_b = tif.pages[0].asarray()

        mask_a = self._get_mask(image=img_a)
        mask_b = self._get_mask(image=img_b)

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
    def __init__(self, root,count = None,transforms_=None, unaligned=False):
        self.transform = Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

        model = SamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self._sam_model = model
        self._sam_processor = processor

    def _get_mask(self, image: np.ndarray):
        raw_image_3channel = np.stack([image, image, image], axis=-1)
        input_boxes = [[[0, 0, image.shape[1], image.shape[0]]]]
        inputs = self._sam_processor(raw_image_3channel, input_boxes=[input_boxes], return_tensors="pt")
        with torch.no_grad():
            outputs = self._sam_model(**inputs, multimask_output=False)
        masks = self._sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(),
                                                             inputs["original_sizes"].cpu(),
                                                             inputs["reshaped_input_sizes"].cpu())
        mask = masks[0][0, 0].numpy().astype('uint8')
        return mask
        
    def __getitem__(self, index):
        with tifffile.TiffReader(self.files_A[index % len(self.files_A)]) as tif:
            img_a = tif.pages[0].asarray()
        with tifffile.TiffReader(self.files_B[index % len(self.files_A)]) as tif:
            img_b = tif.pages[0].asarray()

        mask_a = self._get_mask(image=img_a)
        mask_b = self._get_mask(image=img_b)

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
        return max(len(self.files_A), len(self.files_B))
