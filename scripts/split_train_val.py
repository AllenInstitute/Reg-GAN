import random
import shutil
from pathlib import Path

import click
from loguru import logger


def index_from_filename(path: Path) -> str:
    return path.name.split("_", 1)[0]


def collect_indices(dir_path: Path, suffix: str) -> dict[str, Path]:
    files = {}
    for p in dir_path.iterdir():
        if not p.is_file() or not p.name.endswith(suffix):
            continue
        files[index_from_filename(p)] = p
    return files


@click.command()
@click.option("--input-dir", required=True, type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--train-frac", default=0.8, type=float, show_default=True)
@click.option("--seed", default=42, type=int, show_default=True)
@click.option("--a-suffix", default="_other.tiff", show_default=True,
              help="Filename suffix for A (paired with --b-suffix on the same index).")
@click.option("--b-suffix", default="_light_sheet.tiff", show_default=True)
@click.option("--copy/--move", default=False, help="Copy files instead of moving them.")
def main(input_dir: Path, output_dir: Path, train_frac: float, seed: int,
         a_suffix: str, b_suffix: str, copy: bool):
    if not 0 < train_frac < 1:
        raise click.BadParameter("--train-frac must be in (0, 1)")

    a_files = collect_indices(input_dir, a_suffix)
    b_files = collect_indices(input_dir, b_suffix)

    common = sorted(set(a_files) & set(b_files))
    missing_b = set(a_files) - set(b_files)
    missing_a = set(b_files) - set(a_files)
    if missing_a:
        logger.warning("{} indices have {} but no {}: {}", len(missing_a), b_suffix, a_suffix, sorted(missing_a)[:5])
    if missing_b:
        logger.warning("{} indices have {} but no {}: {}", len(missing_b), a_suffix, b_suffix, sorted(missing_b)[:5])
    if not common:
        raise click.ClickException(f"No matching indices in {input_dir} for suffixes {a_suffix!r} / {b_suffix!r}.")

    rng = random.Random(seed)
    indices = list(common)
    rng.shuffle(indices)
    n_train = int(round(len(indices) * train_frac))
    train_idx = set(indices[:n_train])
    val_idx = set(indices[n_train:])
    logger.info("Total: {} | train: {} | val: {}", len(indices), len(train_idx), len(val_idx))

    op = shutil.copy2 if copy else shutil.move
    for split, idx_set in (("train", train_idx), ("val", val_idx)):
        for sub in ("A", "B"):
            (output_dir / split / sub).mkdir(parents=True, exist_ok=True)
        for idx in sorted(idx_set):
            for sub, src in (("A", a_files[idx]), ("B", b_files[idx])):
                op(str(src), str(output_dir / split / sub / src.name))
        logger.info("{}: wrote {} pairs to {}", split, len(idx_set), output_dir / split)


if __name__ == "__main__":
    main()
