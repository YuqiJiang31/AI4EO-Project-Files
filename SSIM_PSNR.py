import os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

CONFIG = {
    "GT_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Truth",
    "SR_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Bicubic",
    "OUT_DIR": r"E:\Text\AI4EOFINAL\SSIM_PSNR_RESULT",
    "BATCH": 1,
    "EXTENSIONS": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    "FORCE_CPU": False,
    "SKIP_MISMATCH": False,
    "FOLDER_LEVEL_ONLY": False,
    "VERBOSE": True,
    "SR_SUFFIX": "",      # additional suffix compared to GT leave blank if not provided
    "CONVERT_TO_Y": False,
}

def log(msg: str):
    print(msg)

def collect_pairs(gt_root: Path, sr_root: Path, exts: Tuple[str, ...], sr_suffix: str = "", verbose=False) -> List[Tuple[Path, Path, str]]:
    pairs = []
    for dirpath, _, filenames in os.walk(gt_root):
        rel_dir = os.path.relpath(dirpath, gt_root)
        sr_dir_equiv = sr_root / rel_dir if rel_dir != "." else sr_root
        if not sr_dir_equiv.exists():
            if verbose:
                log(f"[Missing SR subdir] {sr_dir_equiv}")
            continue
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() not in exts:
                continue
            gt_p = Path(dirpath) / fn
            sr_p = sr_dir_equiv / fn
            chosen_sr = None
            if sr_p.exists():
                chosen_sr = sr_p
            elif sr_suffix:
                stem, ext = os.path.splitext(fn)
                cand = sr_dir_equiv / f"{stem}{sr_suffix}{ext}"
                if cand.exists():
                    chosen_sr = cand
                    if verbose:
                        log(f"[Suffix match] {fn} -> {cand.name}")
            if chosen_sr:
                pairs.append((gt_p, chosen_sr, "" if rel_dir == "." else rel_dir))
            else:
                if verbose:
                    log(f"[Missing SR image] {sr_dir_equiv / fn} (also not found with suffix {sr_suffix})")
    if verbose:
        log(f"[Debug] Collected pair count: {len(pairs)}")
    return pairs

def load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)

def to_y_channel(img: np.ndarray) -> np.ndarray:
    # img: H W 3, 0-255 uint8
    return (0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]).astype(img.dtype)

def compute_metrics(gt: np.ndarray, sr: np.ndarray, use_y: bool) -> Tuple[float, float]:
    if use_y:
        gt_proc = to_y_channel(gt)
        sr_proc = to_y_channel(sr)
        channel_axis = None
    else:
        gt_proc = gt
        sr_proc = sr
        channel_axis = -1 if gt_proc.ndim == 3 and gt_proc.shape[2] in (1,3,4) else None

    # Normalize dtype if different
    if gt_proc.dtype != sr_proc.dtype:
        # Convert to float32
        gt_proc = gt_proc.astype(np.float32)
        sr_proc = sr_proc.astype(np.float32)

    # Determine data range
    if gt_proc.dtype == np.uint8:
        data_range = 255
    else:
        # If values exceed [0,1], infer range from actual min/max
        max_val = max(float(np.max(gt_proc)), float(np.max(sr_proc)))
        min_val = min(float(np.min(gt_proc)), float(np.min(sr_proc)))
        data_range = max_val - min_val if max_val > min_val else 1.0

    # SSIM
    try:
        ssim_val = structural_similarity(
            gt_proc, sr_proc,
            data_range=data_range,
            channel_axis=channel_axis
        )
    except TypeError:
        multichannel = channel_axis is not None
        ssim_val = structural_similarity(
            gt_proc, sr_proc,
            data_range=data_range,
            multichannel=multichannel
        )

    # PSNR
    psnr_val = peak_signal_noise_ratio(gt_proc, sr_proc, data_range=data_range)
    return float(ssim_val), float(psnr_val)

def main():
    cfg = CONFIG
    gt_root = Path(cfg["GT_DIR"])
    sr_root = Path(cfg["SR_DIR"])
    out_dir = Path(cfg["OUT_DIR"])

    if cfg["VERBOSE"]:
        log(f"[Paths] GT={gt_root}  SR={sr_root}")
    if not gt_root.exists() or not sr_root.exists():
        log("[Error] GT or SR directory does not exist.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(
        gt_root,
        sr_root,
        tuple(e.lower() for e in cfg["EXTENSIONS"]),
        sr_suffix=cfg.get("SR_SUFFIX", ""),
        verbose=cfg["VERBOSE"]
    )
    if not pairs:
        log("No matched image pairs found.")
        return

    log(f"[Progress] Found {len(pairs)} pairs, start computing SSIM / PSNR ...")

    folder_groups = defaultdict(list)
    for g, s, f in pairs:
        folder_groups[f].append((g, s))

    per_image = []
    for folder, lst in folder_groups.items():
        log(f"\n[Folder] {folder if folder else '(root)'} total {len(lst)} images")
        for gt_p, sr_p in tqdm(lst, desc=f"Compute {folder if folder else 'root'}"):
            try:
                gt_img = load_image(gt_p)
                sr_img = load_image(sr_p)
            except Exception as e:
                log(f"[Read failed] {gt_p} / {sr_p}: {e}")
                continue
            if gt_img.shape != sr_img.shape:
                msg = f"[Shape mismatch] {gt_p.name} GT{gt_img.shape} vs SR{sr_img.shape}"
                if cfg["SKIP_MISMATCH"]:
                    log(msg + " -> skip")
                    continue
                else:
                    log(msg)
                    return
            try:
                ssim_v, psnr_v = compute_metrics(gt_img, sr_img, cfg["CONVERT_TO_Y"])
            except Exception as e:
                log(f"[Compute failed] {gt_p.name}: {e}")
                continue
            per_image.append({
                "folder": folder,
                "filename": gt_p.name,
                "ssim": ssim_v,
                "psnr": psnr_v,
                "gt_path": str(gt_p),
                "sr_path": str(sr_p)
            })

    if not per_image:
        log("[Result] No successful entries.")
        return

    df = pd.DataFrame(per_image)
    folder_stats = (
        df.groupby("folder")[["ssim","psnr"]]
        .agg(["count","mean"])
    )
    # Flatten multi-index columns
    folder_stats.columns = ['_'.join(col).rstrip('_') for col in folder_stats.columns.values]
    folder_stats = folder_stats.reset_index()

    overall_ssim = df["ssim"].mean()
    overall_psnr = df["psnr"].mean()

    if not cfg["FOLDER_LEVEL_ONLY"]:
        df.to_csv(out_dir / "ssim_psnr_per_image.csv", index=False, encoding="utf-8-sig")
    folder_stats.to_csv(out_dir / "ssim_psnr_folder_mean.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "ssim_psnr_overall.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall SSIM mean: {overall_ssim:.6f}\n")
        f.write(f"Overall PSNR mean: {overall_psnr:.6f}\n")
        f.write(f"Use_Y={cfg['CONVERT_TO_Y']}\n")

    log("\n[Summary]")
    log(str(folder_stats))
    log(f"[Overall] SSIM mean: {overall_ssim:.6f}")
    log(f"[Overall] PSNR mean: {overall_psnr:.6f}")
    log(f"[Output dir] {out_dir}")

if __name__ == "__main__":
    main()