<<<<<<< HEAD
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def is_image(p: Path, exts):
    return p.is_file() and p.suffix.lower() in exts

def process_image(in_path: Path, out_path: Path, scale: int):
    try:
        with Image.open(in_path) as im:
            w, h = im.size
            new_size = (w * scale, h * scale)
            # bicubic interpolation
            im_up = im.resize(new_size, Image.BICUBIC)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im_up.save(out_path)
    except Exception as e:
        print(f"Skip {in_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch 4x super-resolution using bicubic interpolation")
    parser.add_argument("--input", "-i", required=True, help="Input root directory (with subfolders)")
    parser.add_argument("--output", "-o", required=True, help="Output root directory")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Scale factor, default 4")
    parser.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"], help="Allowed image extensions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    exts = {e.lower() for e in args.ext}

    if not in_root.exists():
        print("Input directory does not exist")
        return

    images = [p for p in in_root.rglob("*") if is_image(p, exts)]
    if not images:
        print("No images found")
        return

    for img_path in tqdm(images, desc="Processing", unit="img"):
        rel = img_path.relative_to(in_root)
        out_path = out_root / rel
        if out_path.exists() and not args.overwrite:
            continue
        process_image(img_path, out_path, args.scale)

    print("Done")

if __name__ == "__main__":
=======
import os
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def is_image(p: Path, exts):
    return p.is_file() and p.suffix.lower() in exts

def process_image(in_path: Path, out_path: Path, scale: int):
    try:
        with Image.open(in_path) as im:
            w, h = im.size
            new_size = (w * scale, h * scale)
            # bicubic interpolation
            im_up = im.resize(new_size, Image.BICUBIC)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            im_up.save(out_path)
    except Exception as e:
        print(f"Skip {in_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Batch 4x super-resolution using bicubic interpolation")
    parser.add_argument("--input", "-i", required=True, help="Input root directory (with subfolders)")
    parser.add_argument("--output", "-o", required=True, help="Output root directory")
    parser.add_argument("--scale", "-s", type=int, default=4, help="Scale factor, default 4")
    parser.add_argument("--ext", nargs="+", default=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"], help="Allowed image extensions")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    exts = {e.lower() for e in args.ext}

    if not in_root.exists():
        print("Input directory does not exist")
        return

    images = [p for p in in_root.rglob("*") if is_image(p, exts)]
    if not images:
        print("No images found")
        return

    for img_path in tqdm(images, desc="Processing", unit="img"):
        rel = img_path.relative_to(in_root)
        out_path = out_root / rel
        if out_path.exists() and not args.overwrite:
            continue
        process_image(img_path, out_path, args.scale)

    print("Done")

if __name__ == "__main__":
>>>>>>> 7c2f2f8742e4ed38e944bf84dbd72169c58a4bf0
    main()