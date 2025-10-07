<<<<<<< HEAD
import os
from pathlib import Path
from typing import List, Tuple
import torch
import lpips
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as TF
from collections import defaultdict

CONFIG = {
    "GT_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Truth",
    "SR_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_SwinIR",
    "OUT_DIR": r"E:\Text\AI4EOFINAL\LPIPS_RESULT",
    "NET": "vgg",
    "BATCH": 8,
    "EXTENSIONS": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    "FORCE_CPU": False,
    "SKIP_MISMATCH": False,
    "FOLDER_LEVEL_ONLY": False,
    "VERBOSE": True,
    "SR_SUFFIX": "_SwinIR"          # Added: extra suffix in SR filenames relative to GT; leave "" if none
}

def log(msg: str):
    print(msg)

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)        # [0,1]
    t = t * 2 - 1                # [-1,1]
    return t

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
            sr_p = sr_dir_equiv / fn  # try same name first
            chosen_sr = None
            if sr_p.exists():
                chosen_sr = sr_p
            elif sr_suffix:  # try with suffix
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
        log(f"[Debug] Collected pairs: {len(pairs)}")
    return pairs

def check_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape[-2:] == b.shape[-2:]

def run_batch(loss_fn, gts, srs, device):
    with torch.no_grad():
        g = torch.stack(gts).to(device)
        s = torch.stack(srs).to(device
        )
        v = loss_fn(g, s)  # (N,1,1,1)
        return v.view(-1).cpu().tolist()

def main():
    cfg = CONFIG
    gt_root = Path(cfg["GT_DIR"])
    sr_root = Path(cfg["SR_DIR"])
    out_dir = Path(cfg["OUT_DIR"])

    if cfg["VERBOSE"]:
        log(f"[Paths] GT={gt_root}  SR={sr_root}")
    if not gt_root.exists() or not sr_root.exists():
        log("[Error] GT or SR directory does not exist. Please check CONFIG.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if cfg["FORCE_CPU"] or not torch.cuda.is_available() else "cuda")
    log(f"[Device] {device}")

    try:
        loss_fn = lpips.LPIPS(net=cfg["NET"]).to(device).eval()
    except Exception as e:
        log(f"[Error] Failed to load LPIPS model: {e}")
        return
    log(f"[Model] LPIPS({cfg['NET']}) loaded")

    pairs = collect_pairs(
        gt_root,
        sr_root,
        tuple(e.lower() for e in cfg["EXTENSIONS"]),
        sr_suffix=cfg.get("SR_SUFFIX", ""),
        verbose=cfg["VERBOSE"]
    )
    if not pairs:
        log("No matched image pairs found.")
        if cfg["VERBOSE"]:
            gt_dirs = [p.name for p in gt_root.iterdir() if p.is_dir()]
            sr_dirs = [p.name for p in sr_root.iterdir() if p.is_dir()]
            log(f"[GT subdirs]{gt_dirs[:20]}")
            log(f"[SR subdirs]{sr_dirs[:20]}")
        return

    log(f"[Progress] Found {len(pairs)} pairs, start computing...")

    folder_groups = defaultdict(list)
    for g, s, f in pairs:
        folder_groups[f].append((g, s))

    per_image = []
    for folder, lst in folder_groups.items():
        log(f"\n[Folder] {folder if folder else '(root)'} total {len(lst)} images")
        batch_gt, batch_sr, batch_meta = [], [], []
        for gt_p, sr_p in tqdm(lst, desc=f"Compute {folder if folder else 'root'}"):
            try:
                g_img = load_image_tensor(gt_p)
                s_img = load_image_tensor(sr_p)
            except Exception as e:
                log(f"[Read failed] {gt_p} / {sr_p}: {e}")
                continue
            if not check_match(g_img, s_img):
                msg = f"[Size mismatch] {gt_p.name} GT{g_img.shape[-2:]} vs SR{s_img.shape[-2:]}"
                if cfg["SKIP_MISMATCH"]:
                    log(msg + " -> skip")
                    continue
                else:
                    raise ValueError(msg)
            batch_gt.append(g_img)
            batch_sr.append(s_img)
            batch_meta.append((folder, gt_p.name, str(gt_p), str(sr_p)))
            if len(batch_gt) == cfg["BATCH"]:
                vals = run_batch(loss_fn, batch_gt, batch_sr, device)
                for meta, v in zip(batch_meta, vals):
                    ftag, fname, gpath, spath = meta
                    per_image.append({
                        "folder": ftag,
                        "filename": fname,
                        "lpips": float(v),
                        "gt_path": gpath,
                        "sr_path": spath
                    })
                batch_gt, batch_sr, batch_meta = [], [], []
        if batch_gt:
            vals = run_batch(loss_fn, batch_gt, batch_sr, device)
            for meta, v in zip(batch_meta, vals):
                ftag, fname, gpath, spath = meta
                per_image.append({
                    "folder": ftag,
                    "filename": fname,
                    "lpips": float(v),
                    "gt_path": gpath,
                    "sr_path": spath
                })

    if not per_image:
        log("[Result] No successful entries.")
        return

    df = pd.DataFrame(per_image)
    folder_stats = (
        df.groupby("folder")["lpips"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "lpips_mean"})
    )
    overall_mean = df["lpips"].mean()

    if not cfg["FOLDER_LEVEL_ONLY"]:
        df.to_csv(out_dir / "lpips_per_image.csv", index=False, encoding="utf-8-sig")
    folder_stats.to_csv(out_dir / "lpips_folder_mean.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "lpips_overall.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall LPIPS mean ({cfg['NET']}): {overall_mean:.6f}\n")

    log("\n[Summary]")
    log(str(folder_stats))
    log(f"[Overall] LPIPS mean ({cfg['NET']}): {overall_mean:.6f}")
    log(f"[Output dir] {out_dir}")

if __name__ == "__main__":
=======
import os
from pathlib import Path
from typing import List, Tuple
import torch
import lpips
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torchvision.transforms.functional as TF
from collections import defaultdict

CONFIG = {
    "GT_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_Truth",
    "SR_DIR": r"E:\Text\AI4EOFINAL\UCMerced_HR_SwinIR",
    "OUT_DIR": r"E:\Text\AI4EOFINAL\LPIPS_RESULT",
    "NET": "vgg",
    "BATCH": 8,
    "EXTENSIONS": (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    "FORCE_CPU": False,
    "SKIP_MISMATCH": False,
    "FOLDER_LEVEL_ONLY": False,
    "VERBOSE": True,
    "SR_SUFFIX": "_SwinIR"          # Added: extra suffix in SR filenames relative to GT; leave "" if none
}

def log(msg: str):
    print(msg)

def load_image_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = TF.to_tensor(img)        # [0,1]
    t = t * 2 - 1                # [-1,1]
    return t

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
            sr_p = sr_dir_equiv / fn  # try same name first
            chosen_sr = None
            if sr_p.exists():
                chosen_sr = sr_p
            elif sr_suffix:  # try with suffix
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
        log(f"[Debug] Collected pairs: {len(pairs)}")
    return pairs

def check_match(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape[-2:] == b.shape[-2:]

def run_batch(loss_fn, gts, srs, device):
    with torch.no_grad():
        g = torch.stack(gts).to(device)
        s = torch.stack(srs).to(device
        )
        v = loss_fn(g, s)  # (N,1,1,1)
        return v.view(-1).cpu().tolist()

def main():
    cfg = CONFIG
    gt_root = Path(cfg["GT_DIR"])
    sr_root = Path(cfg["SR_DIR"])
    out_dir = Path(cfg["OUT_DIR"])

    if cfg["VERBOSE"]:
        log(f"[Paths] GT={gt_root}  SR={sr_root}")
    if not gt_root.exists() or not sr_root.exists():
        log("[Error] GT or SR directory does not exist. Please check CONFIG.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if cfg["FORCE_CPU"] or not torch.cuda.is_available() else "cuda")
    log(f"[Device] {device}")

    try:
        loss_fn = lpips.LPIPS(net=cfg["NET"]).to(device).eval()
    except Exception as e:
        log(f"[Error] Failed to load LPIPS model: {e}")
        return
    log(f"[Model] LPIPS({cfg['NET']}) loaded")

    pairs = collect_pairs(
        gt_root,
        sr_root,
        tuple(e.lower() for e in cfg["EXTENSIONS"]),
        sr_suffix=cfg.get("SR_SUFFIX", ""),
        verbose=cfg["VERBOSE"]
    )
    if not pairs:
        log("No matched image pairs found.")
        if cfg["VERBOSE"]:
            gt_dirs = [p.name for p in gt_root.iterdir() if p.is_dir()]
            sr_dirs = [p.name for p in sr_root.iterdir() if p.is_dir()]
            log(f"[GT subdirs]{gt_dirs[:20]}")
            log(f"[SR subdirs]{sr_dirs[:20]}")
        return

    log(f"[Progress] Found {len(pairs)} pairs, start computing...")

    folder_groups = defaultdict(list)
    for g, s, f in pairs:
        folder_groups[f].append((g, s))

    per_image = []
    for folder, lst in folder_groups.items():
        log(f"\n[Folder] {folder if folder else '(root)'} total {len(lst)} images")
        batch_gt, batch_sr, batch_meta = [], [], []
        for gt_p, sr_p in tqdm(lst, desc=f"Compute {folder if folder else 'root'}"):
            try:
                g_img = load_image_tensor(gt_p)
                s_img = load_image_tensor(sr_p)
            except Exception as e:
                log(f"[Read failed] {gt_p} / {sr_p}: {e}")
                continue
            if not check_match(g_img, s_img):
                msg = f"[Size mismatch] {gt_p.name} GT{g_img.shape[-2:]} vs SR{s_img.shape[-2:]}"
                if cfg["SKIP_MISMATCH"]:
                    log(msg + " -> skip")
                    continue
                else:
                    raise ValueError(msg)
            batch_gt.append(g_img)
            batch_sr.append(s_img)
            batch_meta.append((folder, gt_p.name, str(gt_p), str(sr_p)))
            if len(batch_gt) == cfg["BATCH"]:
                vals = run_batch(loss_fn, batch_gt, batch_sr, device)
                for meta, v in zip(batch_meta, vals):
                    ftag, fname, gpath, spath = meta
                    per_image.append({
                        "folder": ftag,
                        "filename": fname,
                        "lpips": float(v),
                        "gt_path": gpath,
                        "sr_path": spath
                    })
                batch_gt, batch_sr, batch_meta = [], [], []
        if batch_gt:
            vals = run_batch(loss_fn, batch_gt, batch_sr, device)
            for meta, v in zip(batch_meta, vals):
                ftag, fname, gpath, spath = meta
                per_image.append({
                    "folder": ftag,
                    "filename": fname,
                    "lpips": float(v),
                    "gt_path": gpath,
                    "sr_path": spath
                })

    if not per_image:
        log("[Result] No successful entries.")
        return

    df = pd.DataFrame(per_image)
    folder_stats = (
        df.groupby("folder")["lpips"]
        .agg(["count", "mean"])
        .reset_index()
        .rename(columns={"mean": "lpips_mean"})
    )
    overall_mean = df["lpips"].mean()

    if not cfg["FOLDER_LEVEL_ONLY"]:
        df.to_csv(out_dir / "lpips_per_image.csv", index=False, encoding="utf-8-sig")
    folder_stats.to_csv(out_dir / "lpips_folder_mean.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "lpips_overall.txt", "w", encoding="utf-8") as f:
        f.write(f"Overall LPIPS mean ({cfg['NET']}): {overall_mean:.6f}\n")

    log("\n[Summary]")
    log(str(folder_stats))
    log(f"[Overall] LPIPS mean ({cfg['NET']}): {overall_mean:.6f}")
    log(f"[Output dir] {out_dir}")

if __name__ == "__main__":
>>>>>>> 7c2f2f8742e4ed38e944bf84dbd72169c58a4bf0
    main()