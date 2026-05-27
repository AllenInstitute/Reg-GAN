import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import random
import matplotlib.pyplot as plt
from skimage import measure
from scipy import ndimage
import torch

from train_segmentation_model import TissueMaskModel


def largest_component(mask: np.ndarray) -> np.ndarray:
    labels, label_count = ndimage.label(mask)
    if label_count == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def _prepare_image(image: np.ndarray, multiple: int = 32) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D TIFF image, got shape {image.shape}")

    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    image = image.astype(np.float32)
    image = image / 255.0
    image = (image - 0.5) / 0.5
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return tensor, (h, w)


def _get_mask(image: np.ndarray, model: TissueMaskModel, device, threshold: float) -> np.ndarray:
    tensor, (h, w) = _prepare_image(image)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        mask = torch.sigmoid(logits)[0, 0, :h, :w] > threshold
    mask = largest_component(mask.cpu().numpy())
    mask = ndimage.binary_fill_holes(mask)
    return mask.astype("uint8")


def display_masks():
    random.seed(0)

    root = Path('/results/slice_registration_style_transfrer_dataset')
    mask_root = Path('/results/masks')
    out_path = Path('/scratch/fig.png')

    files = os.listdir(mask_root)
    files = [x for x in files if Path(x).suffix.lower() == '.png']

    light_sheet_files = sorted([x for x in files if Path(x).stem.endswith('light_sheet_template_mask')])
    other_files = sorted([x for x in files if Path(x).stem.endswith('other_mask')])

    sampled = []
    for label, group in [('light_sheet_template', light_sheet_files), ('other', other_files)]:
        if len(group) < 10:
            raise ValueError(f'Only found {len(group)} {label} files; need 10')
        sampled.extend((label, f) for f in random.sample(group, 10))

    fig, axes = plt.subplots(4, 5, figsize=(20, 16), constrained_layout=True)
    axes = axes.ravel()

    for ax, (label, file) in zip(axes, sampled):
        mask_path = mask_root / file
        fname = Path(file).stem.removesuffix('_mask')
        img_path = root / f'{fname}.tiff'

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        if not img_path.exists():
            raise FileNotFoundError(img_path)

        image = tifffile.imread(img_path)
        if image.ndim > 2:
            image = image[0]

        mask = np.array(Image.open(mask_path)) > 0

        ax.imshow(image, cmap='gray')
        for contour in measure.find_contours(mask.astype(float), 0.5):
            ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1.0)

        ax.set_title(f'{fname}\n{label}', fontsize=9)
        ax.axis('off')

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def main(checkpoint_path: Path, input_dir: Path, output_dir: Path, threshold: float, limit: int | None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TissueMaskModel.load_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    files = sorted(x for x in os.listdir(input_dir) if Path(x).suffix == '.tiff')
    if limit is not None:
        files = files[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    for file in tqdm(files):
        image = tifffile.imread(input_dir / file)
        mask = _get_mask(image=image, model=model, device=device, threshold=threshold)
        fname = Path(file).stem
        Image.fromarray(mask).save(output_dir / f'{fname}_mask.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true', help='Write sampled generated masks figure to /scratch/fig.png.')
    parser.add_argument('--checkpoint-path', type=Path, required=False, help='Lightning checkpoint from train_segmentation_model.py.')
    parser.add_argument('--input-dir', type=Path, default=Path('/results/slice_registration_style_transfrer_dataset'))
    parser.add_argument('--output-dir', type=Path, default=Path('/results/masks'))
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    if args.display:
        display_masks()
    else:
        if args.checkpoint_path is None:
            parser.error('--checkpoint-path is required unless --display is used')
        main(
            checkpoint_path=args.checkpoint_path,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            threshold=args.threshold,
            limit=args.limit,
        )
