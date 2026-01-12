# ============================================================
# SHAPExplainer (all-in-one)
#
# What it does INSIDE the class:
# 1) Finds stations
# 2) Ensures trained_models_correct_test_samples/<STATION> exists
#    (creates it by running inference on test set and copying correct pngs)
# 3) Computes SHAP tensors per trace and saves them
# 4) Computes mean SHAP maps and saves them
# 5) Plots:
#    - one station (pre OR post)
#    - one station (both pre+post)
#    - many stations (only pre OR only post)
#    - many stations (both pre+post)
#
# Supports:
# - trained_models_dir variable (e.g. "trained_models/models")
# - preprocessed_dir variable (e.g. "preprocessed_dset1")
# - auto station discovery
# - SHAP backend:
#     * modern SHAP: shap.maskers + shap.Explainer (if available)
#     * legacy fallback: shap.GradientExplainer (if maskers missing)
#
# IMPORTANT:
# - Do NOT name this file shap.py (it will shadow the shap package).
#   Name it shap_explainer.py or similar.
# ============================================================

import json
import gc
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import shap  
from tqdm import tqdm
from training import CNN2D  
from pathlib import Path

def get_project_root(marker="preprocessed_dset1"):
    here = Path.cwd().resolve()
    for p in [here] + list(here.parents):
        if (p / marker).exists():
            return p
    raise RuntimeError("Project root not found")

def ensure_2d(arr):
    a = np.asarray(arr)

    while a.ndim >= 3 and a.shape[0] == 1:
        a = np.squeeze(a, axis=0)

    if a.ndim == 4 and a.shape[-1] in (1, 2) and a.shape[-2] in (3, 4):
        a = a[..., 0]

    if a.ndim == 3 and a.shape[-1] in (3, 4):
        a = a.mean(axis=-1)

    if a.ndim != 2:
        raise ValueError(f"Cannot ensure 2D: got {arr.shape} -> {a.shape}")
    return a


def preprocess_for_shap(images):
    min_val = torch.min(images)
    max_val = torch.max(images)
    x = (images - min_val) / (max_val - min_val + 1e-12)
    x = (x * 255).to(torch.uint8)
    x = x.permute(0, 2, 3, 1)  # (N,H,W,C)
    return x, float(min_val.item()), float(max_val.item())


def inverse_preprocess_for_shap(shap_images, original_min, original_max):
    x = shap_images.permute(0, 3, 1, 2).float()  # (N,C,H,W)
    x = x / 255.0
    x = x * (original_max - original_min) + original_min
    return x


class PngDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with Image.open(p) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, p.stem

class SHAPExplainer:
    def __init__(
        self,
        trained_models_dir="trained_models/models",
        preprocessed_dir="preprocessed_dset1",
        correct_samples_dir="trained_models_correct_test_samples",
        verbose=True,
    ):
        self.project_root = get_project_root()

        self.models_root = self.project_root / trained_models_dir
        self.pre_root = self.project_root / preprocessed_dir
        self.correct_root = self.project_root / correct_samples_dir

        self.verbose = bool(verbose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.has_modern_shap = hasattr(shap, "maskers") and hasattr(shap, "Explainer")
        if self.verbose:
            print("[SHAPExplainer]")
            print("  project_root   :", self.project_root)
            print("  pre_root       :", self.pre_root)
            print("  models_root    :", self.models_root)
            print("  correct_root   :", self.correct_root)
            print("  device         :", self.device)
            print("  modern_shap?   :", self.has_modern_shap)

    def list_stations(self):
        if not self.pre_root.exists():
            return []
        stations = []
        for p in sorted(self.pre_root.iterdir()):
            if p.is_dir() and (p / "input_dim.npy").exists() and (p / "test").exists():
                stations.append(p.name)
        return stations

    def _station_dirs(self, station):
        pre_dir = self.pre_root / station
        model_dir = self.models_root / station
        correct_dir = self.correct_root / station
        return pre_dir, model_dir, correct_dir

    def _load_input_dim(self, pre_dir):
        h, w = np.load(pre_dir / "input_dim.npy").astype(int).tolist()
        return (int(h), int(w))

    def _load_stats(self, model_dir):
        d = json.loads((model_dir / "stats.json").read_text())
        return d["mean"], d["std"]

    def _load_model(self, station):
        pre_dir, model_dir, correct_dir = self._station_dirs(station)

        ckpt = model_dir / "best.ckpt"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not (model_dir / "stats.json").exists():
            raise FileNotFoundError(f"Missing stats.json: {model_dir / 'stats.json'}")
        if not (pre_dir / "input_dim.npy").exists():
            raise FileNotFoundError(f"Missing input_dim.npy: {pre_dir / 'input_dim.npy'}")

        input_dim = self._load_input_dim(pre_dir)
        mean, std = self._load_stats(model_dir)

        model = CNN2D.load_from_checkpoint(
            str(ckpt),
            input_dim=input_dim,
            n_classes=2,
            lr=1e-3,
            dropout=0.1
        )
        model.eval()
        model.to(self.device)

        return model, input_dim, mean, std, pre_dir, model_dir, correct_dir

    def _load_ft(self, station):
        pre_dir, _, _ = self._station_dirs(station)
        p = pre_dir / "f_t_range.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing f_t_range.npy: {p}")
        tf = np.load(p).astype(float)
        return [tf[:2], tf[2:]]

    def ensure_correct_samples(self, station, max_per_class=100, overwrite=False):
        pre_dir, model_dir, correct_dir = self._station_dirs(station)

        out_fo = correct_dir / "foreshock"
        out_af = correct_dir / "aftershock"

        if correct_dir.exists() and not overwrite:
            out_fo.mkdir(parents=True, exist_ok=True)
            out_af.mkdir(parents=True, exist_ok=True)
            return

        out_fo.mkdir(parents=True, exist_ok=True)
        out_af.mkdir(parents=True, exist_ok=True)

        if overwrite:
            for d in [out_fo, out_af]:
                for p in d.glob("*.png"):
                    try:
                        p.unlink()
                    except Exception:
                        pass

        model, input_dim, mean, std, pre_dir, _, _ = self._load_model(station)

        tfm = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        test_root = pre_dir / "test"
        fo_in = test_root / "foreshock"
        af_in = test_root / "aftershock"

        saved = {"foreshock": 0, "aftershock": 0}

        def copy_if_correct(png_path, true_label_int, out_folder):
            nonlocal saved
            with Image.open(png_path) as im:
                im = im.convert("RGB")
                x = tfm(im).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model(x)
                pred = int(torch.argmax(logits, dim=1).item())
            if pred != true_label_int:
                return
            dst = out_folder / png_path.name
            dst.write_bytes(png_path.read_bytes())
            if true_label_int == 0:
                saved["foreshock"] += 1
            else:
                saved["aftershock"] += 1

        if fo_in.exists():
            for p in sorted(fo_in.glob("*.png")):
                if saved["foreshock"] >= int(max_per_class):
                    break
                copy_if_correct(p, 0, out_fo)

        if af_in.exists():
            for p in sorted(af_in.glob("*.png")):
                if saved["aftershock"] >= int(max_per_class):
                    break
                copy_if_correct(p, 1, out_af)

        if self.verbose:
            print(f"[OK] {station} correct samples saved -> {correct_dir} | {saved}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _compute_shap_one(self, model, x, input_dim, max_evals, batch_size, masker_settings, len_label_channels):
        if self.has_modern_shap:
            images_to_explain, original_min, original_max = preprocess_for_shap(x)

            def model_fn(nhwc_np):
                xx = torch.from_numpy(nhwc_np).to(self.device)
                xx = inverse_preprocess_for_shap(xx, original_min, original_max)
                with torch.no_grad():
                    out = model(xx)
                return out.detach().cpu().numpy()

            masker = shap.maskers.Image(masker_settings, (*input_dim, 3))
            explainer = shap.Explainer(model_fn, masker, output_names=["Foreshock", "Aftershock"])

            sv = explainer(
                images_to_explain.cpu().numpy(),
                max_evals=max_evals,
                batch_size=batch_size,
                outputs=shap.Explanation.argsort.flip[:len_label_channels],
            )
            return np.array(sv.values)

        x = x.to(self.device)
        x.requires_grad = True
        background = torch.zeros_like(x)

        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(x)  # list per output


        sv = shap_values[0]  
        if isinstance(sv, torch.Tensor):
            sv = sv.detach().cpu().numpy()

        sv = sv[0].transpose(1, 2, 0)
        return sv

    def compute_station_shap(
        self,
        station,
        max_per_class=50,
        max_evals=500,
        masker_settings="inpaint_telea",
        len_label_channels=1,
        overwrite=False,
        batch_size_explainer=50,
    ):
        self.ensure_correct_samples(station, max_per_class=max_per_class, overwrite=False)

        model, input_dim, mean, std, pre_dir, model_dir, correct_dir = self._load_model(station)

        tfm = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        def out_dir_for(cls):
            if cls == "foreshock":
                d = model_dir / "shap_tensors" / "foreshocks" / "foreshocks"
            else:
                d = model_dir / "shap_tensors" / "aftershocks" / "aftershocks"
            d.mkdir(parents=True, exist_ok=True)
            return d

        for cls in ["foreshock", "aftershock"]:
            img_dir = correct_dir / cls
            if not img_dir.exists():
                if self.verbose:
                    print(f"[SKIP] {station} {cls}: missing folder {img_dir}")
                continue

            imgs = sorted(img_dir.glob("*.png"))
            if len(imgs) == 0:
                if self.verbose:
                    print(f"[SKIP] {station} {cls}: no pngs in {img_dir}")
                continue

            out_dir = out_dir_for(cls)
            ds = PngDataset(imgs, tfm)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

            if self.verbose:
                print(f"[{station}] SHAP {cls}: n={len(ds)} -> {out_dir}")

            for x, trace_name in tqdm(dl,desc=f"{station} SHAP {cls}",total=len(dl),disable=not self.verbose):
                trace_name = trace_name[0]
                out_path = out_dir / f"{trace_name}.npy"
                if out_path.exists() and not overwrite:
                    continue

                shap_tensor = self._compute_shap_one(
                    model=model,
                    x=x,
                    input_dim=input_dim,
                    max_evals=max_evals,
                    batch_size=batch_size_explainer,
                    masker_settings=masker_settings,
                    len_label_channels=len_label_channels,
                )
                np.save(out_path, shap_tensor)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compute_station_means(self, station, overwrite=True):
        _, _, _, _, _, model_dir, _ = self._load_model(station)

        fo_folder = model_dir / "shap_tensors" / "foreshocks" / "foreshocks"
        af_folder = model_dir / "shap_tensors" / "aftershocks" / "aftershocks"

        def mean_of_folder(folder):
            files = sorted(folder.glob("*.npy"))
            maps = []
            for p in files:
                try:
                    a = np.load(p, allow_pickle=True)
                    m = ensure_2d(a)
                    maps.append(m)
                except Exception:
                    pass
            if len(maps) == 0:
                return None
            shapes = {m.shape for m in maps}
            if len(shapes) != 1:
                raise ValueError(f"Shape mismatch in {folder}: {shapes}")
            return np.mean(np.stack(maps, axis=0), axis=0)

        if fo_folder.exists():
            m = mean_of_folder(fo_folder)
            if m is not None:
                out = model_dir / "shap_tensors" / "foreshocks" / "avg_p_pre.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not out.exists():
                    np.save(out, m)
                if self.verbose:
                    print(f"[OK] {station} saved {out} shape={m.shape}")
        if af_folder.exists():
            m = mean_of_folder(af_folder)
            if m is not None:
                out = model_dir / "shap_tensors" / "aftershocks" / "avg_p_post.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                if overwrite or not out.exists():
                    np.save(out, m)
                if self.verbose:
                    print(f"[OK] {station} saved {out} shape={m.shape}")

    def compute_all(
        self,
        stations=None,
        max_per_class=50,
        max_evals=500,
        overwrite_shap=False,
        overwrite_means=True,
    ):
        if stations is None:
            stations = self.list_stations()

        for st in stations:
            try:
                if self.verbose:
                    print("\n==============================")
                    print("STATION:", st)
                    print("==============================")
                self.compute_station_shap(
                    st,
                    max_per_class=max_per_class,
                    max_evals=max_evals,
                    overwrite=overwrite_shap
                )
                self.compute_station_means(st, overwrite=overwrite_means)
            except Exception as e:
                print(f"[ERROR] {st}: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _load_mean_map(self, station, which):
        _, _, _, _, _, model_dir, _ = self._load_model(station)
        if which == "pre":
            p = model_dir / "shap_tensors" / "foreshocks" / "avg_p_pre.npy"
        else:
            p = model_dir / "shap_tensors" / "aftershocks" / "avg_p_post.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing mean map: {p}")
        return ensure_2d(np.load(p, allow_pickle=True))

    def _imshow_map(self, ax, m, ft, vmin, vmax, alpha_norm, title, show_ylabel):
        f, t = ft
        ax.set_title(title)
        ax.axvline(5.0, c="black", lw=2, alpha=0.3)
        im = ax.imshow(
            m,
            cmap="coolwarm",
            alpha=np.clip(np.abs(m) / alpha_norm, 0, 1),
            aspect="auto",
            origin="lower",
            extent=[*t, *f],
            vmin=vmin,
            vmax=vmax,
        )
        if show_ylabel:
            ax.set_ylabel("Frequency [Hz]")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Time [s]")
        return im

    def plot_station_single(self, station, which, ft=None, symmetric_cbar=True, figsize=(14, 5)):
        if ft is None:
            ft = self._load_ft(station)
        m = self._load_mean_map(station, which)

        vmin, vmax = float(m.min()), float(m.max())
        if symmetric_cbar:
            mm = max(abs(vmin), abs(vmax))
            vmin, vmax = -mm, mm
        alpha_norm = float(np.max(np.abs(m))) if np.max(np.abs(m)) > 0 else 1.0

        fig, ax = plt.subplots(figsize=figsize)
        im = self._imshow_map(ax, m, ft, vmin, vmax, alpha_norm, f"{station} | {which}", True)
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.1)
        cb.set_label("Contribution")
        plt.show()

    def plot_station_both(self, station, ft=None, symmetric_cbar=True, figsize=(16, 6)):
        if ft is None:
            ft = self._load_ft(station)
        pre = self._load_mean_map(station, "pre")
        post = self._load_mean_map(station, "post")

        vals = np.concatenate([pre.ravel(), post.ravel()])
        vmin, vmax = float(vals.min()), float(vals.max())
        if symmetric_cbar:
            mm = max(abs(vmin), abs(vmax))
            vmin, vmax = -mm, mm
        alpha_norm = float(np.max(np.abs(vals))) if np.max(np.abs(vals)) > 0 else 1.0

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.suptitle(f"{station} | pre vs post")

        im1 = self._imshow_map(axes[0], pre, ft, vmin, vmax, alpha_norm, "pre (foreshock)", True)
        im2 = self._imshow_map(axes[1], post, ft, vmin, vmax, alpha_norm, "post (aftershock)", False)

        cb = plt.colorbar(im2, ax=axes, orientation="horizontal", pad=0.1)
        cb.set_label("Contribution")
        plt.show()

    def plot_stations_single(self, stations, which, ft=None, symmetric_cbar=True, figsize=(24, 6)):
        if ft is None:
            ft = self._load_ft(stations[0])

        maps = []
        for st in stations:
            maps.append(self._load_mean_map(st, which))

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(nrows=2, ncols=len(stations), height_ratios=[1, 0.06])

        for c, st in enumerate(stations):
            m = maps[c]
            vmin, vmax = float(m.min()), float(m.max())
            if symmetric_cbar:
                mm = max(abs(vmin), abs(vmax))
                vmin, vmax = -mm, mm
            alpha_norm = float(np.max(np.abs(m))) if np.max(np.abs(m)) > 0 else 1.0

            ax = fig.add_subplot(gs[0, c])
            ax.set_title(st)
            im = self._imshow_map(ax, m, ft, vmin, vmax, alpha_norm, "", show_ylabel=(c == 0))

            cax = fig.add_subplot(gs[1, c])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=8)

        fig.suptitle(f"Stations | {which}")
        plt.show()

    def plot_stations_both(self, stations, ft=None, symmetric_cbar=True, figsize=(24, 10)):
        if ft is None:
            ft = self._load_ft(stations[0])

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(nrows=3, ncols=len(stations), height_ratios=[1, 1, 0.06])

        for c, st in enumerate(stations):
            pre = self._load_mean_map(st, "pre")
            post = self._load_mean_map(st, "post")

            vals = np.concatenate([pre.ravel(), post.ravel()])
            vmin, vmax = float(vals.min()), float(vals.max())
            if symmetric_cbar:
                mm = max(abs(vmin), abs(vmax))
                vmin, vmax = -mm, mm
            alpha_norm = float(np.max(np.abs(vals))) if np.max(np.abs(vals)) > 0 else 1.0

            ax0 = fig.add_subplot(gs[0, c])
            ax0.set_title(st)
            im0 = self._imshow_map(ax0, pre, ft, vmin, vmax, alpha_norm, "pre", show_ylabel=(c == 0))

            ax1 = fig.add_subplot(gs[1, c])
            im1 = self._imshow_map(ax1, post, ft, vmin, vmax, alpha_norm, "post", show_ylabel=(c == 0))

            cax = fig.add_subplot(gs[2, c])
            cb = fig.colorbar(im1, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=8)

        fig.suptitle("Stations | pre (top) and post (bottom)")
        plt.show()