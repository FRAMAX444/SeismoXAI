import os
import gc
import json
import random
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal
import scipy.stats

class Preprocessing:
    def __init__(
        self,
        n_samples=1300,
        fs=100,
        nperseg=64,
        noverlap_ratio=0.8,
        normalization=True,
        train_frac=0.8,
        force_balance=True,
        seed=42,
        cut_freq=None,  
        eps=1e-10,
        verbose=True
    ):

        self.n_samples = int(n_samples)
        self.fs = int(fs)
        self.nperseg = int(nperseg)
        self.noverlap = int(self.nperseg * float(noverlap_ratio))
        self.normalization = bool(normalization)
        self.train_frac = float(train_frac)
        self.force_balance = bool(force_balance)
        self.seed = int(seed)
        self.cut_freq = cut_freq
        self.eps = float(eps)
        self.verbose = bool(verbose)
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        self.expected_hw = self._expected_hw(self.n_samples)

    def make_dset(self, input_dir, output_dir=None):
        input_dir = Path(input_dir)

        if output_dir is None:
            out_base = input_dir.parent
        else:
            out_base = Path(output_dir)

        out_base.mkdir(parents=True, exist_ok=True)

        # Decide: single station vs root
        station_dirs = []

        if self._has_subfolders(input_dir):
            # Treat as root of stations
            station_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir() and self._looks_like_station_folder(p)])
        else:
            # No subfolders -> treat as single folder dataset
            if self._looks_like_station_folder(input_dir):
                station_dirs = [input_dir]

        if not station_dirs:
            raise FileNotFoundError(
                "No valid station folders found.\n"
                f"Checked: {input_dir}\n"
                "Expected station folder to contain: <STATION>_pre.hdf5 and <STATION>_post.hdf5"
            )

        if self.verbose:
            print(f"Input:  {input_dir}")
            print(f"Output: {out_base}")
            print(f"Stations found: {[p.name for p in station_dirs]}")
            print(f"Expected spectrogram (H, W) = {self.expected_hw}")

        for st_dir in station_dirs:
            st = st_dir.name
            try:
                self._process_station(st_dir=st_dir, out_base=out_base)
            except Exception as e:
                print(f"[ERROR] station {st} failed: {e}")
                gc.collect()

        if self.verbose:
            print("\nDONE. Dataset written to:", out_base)

    def _process_station(self, st_dir, out_base):
        st_dir = Path(st_dir)
        st = st_dir.name

        pre_h5, post_h5, pre_csv, post_csv = self._station_files(st_dir)

        st_out = out_base / st
        self._make_dirs(st_out)

        np.save(st_out / "input_dim.npy", np.array(self.expected_hw, dtype=int))
        with open(st_out / "input_dim.json", "w") as f:
            json.dump({"spec_h": int(self.expected_hw[0]), "spec_w": int(self.expected_hw[1])}, f, indent=2)

        df_pre = self._read_attributes(pre_csv)
        df_post = self._read_attributes(post_csv)

        pre_traces = df_pre["trace_name"].astype(str).tolist()
        post_traces = df_post["trace_name"].astype(str).tolist()

        split = self._split_traces(
            pre_traces=pre_traces,
            post_traces=post_traces,
            train_frac=self.train_frac,
            force_balance=self.force_balance
        )

        if self.verbose:
            print(
                f"\n[{st}] pre={len(pre_traces)} post={len(post_traces)} | "
                f"used_pre={len(split['used_pre'])} used_post={len(split['used_post'])} | "
                f"train/class={len(split['train_pre']) if split['force_balance'] else len(split['train_pre'])} "
                f"test/class={len(split['test_pre']) if split['force_balance'] else len(split['test_pre'])} "
                f"| force_balance={split['force_balance']}"
            )

        df_pre_idx = self._build_index_by_trace(df_pre)
        df_post_idx = self._build_index_by_trace(df_post)

        h_pre = h5py.File(str(pre_h5), "r")
        h_post = h5py.File(str(post_h5), "r")

        meta_rows = []
        station_tf = None

        try:
            # foreshock (pre)
            for trace in tqdm(sorted(split["train_pre"]), desc=f"{st} train foreshock", disable=not self.verbose):
                station_tf, out_row = self._process_one(
                    st=st,
                    trace=trace,
                    row_src=df_pre_idx.get(trace),
                    label_folder="foreshock",
                    split_name="train",
                    hdf=h_pre,
                    st_out=st_out,
                    station_tf=station_tf
                )
                if out_row is not None:
                    meta_rows.append(out_row)

            for trace in tqdm(sorted(split["test_pre"]), desc=f"{st} test foreshock", disable=not self.verbose):
                station_tf, out_row = self._process_one(
                    st=st,
                    trace=trace,
                    row_src=df_pre_idx.get(trace),
                    label_folder="foreshock",
                    split_name="test",
                    hdf=h_pre,
                    st_out=st_out,
                    station_tf=station_tf
                )
                if out_row is not None:
                    meta_rows.append(out_row)

            # aftershock (post)
            for trace in tqdm(sorted(split["train_post"]), desc=f"{st} train aftershock", disable=not self.verbose):
                station_tf, out_row = self._process_one(
                    st=st,
                    trace=trace,
                    row_src=df_post_idx.get(trace),
                    label_folder="aftershock",
                    split_name="train",
                    hdf=h_post,
                    st_out=st_out,
                    station_tf=station_tf
                )
                if out_row is not None:
                    meta_rows.append(out_row)

            for trace in tqdm(sorted(split["test_post"]), desc=f"{st} test aftershock", disable=not self.verbose):
                station_tf, out_row = self._process_one(
                    st=st,
                    trace=trace,
                    row_src=df_post_idx.get(trace),
                    label_folder="aftershock",
                    split_name="test",
                    hdf=h_post,
                    st_out=st_out,
                    station_tf=station_tf
                )
                if out_row is not None:
                    meta_rows.append(out_row)

        finally:
            h_pre.close()
            h_post.close()

        # Write per-station metadata.csv
        st_meta = pd.DataFrame(meta_rows)
        st_meta_path = st_out / "metadata.csv"
        st_meta.to_csv(st_meta_path, index=False)

        if self.verbose:
            print(f"[OK] wrote {st_meta_path} | rows={len(st_meta)}")

        # Cleanup
        del df_pre, df_post, df_pre_idx, df_post_idx, meta_rows, st_meta
        gc.collect()

    def _process_one(self, st, trace, row_src, label_folder, split_name, hdf, st_out, station_tf):
        
        if row_src is None:
            return station_tf, None

        try:
            wf = hdf[trace][:]
        except Exception as e:
            print(f"[SKIP] {st} {label_folder} {trace}: missing in hdf5 ({e})")
            return station_tf, None

        if not isinstance(wf, np.ndarray) or wf.ndim != 2 or wf.shape[0] != 3:
            print(f"[SKIP] {st} {label_folder} {trace}: invalid waveform shape={getattr(wf, 'shape', None)}")
            return station_tf, None

        # Crop/Pad to n_samples (you requested consistent length; here we crop; pad if shorter)
        wf = self._fix_length(wf, self.n_samples)

        # Compute spectrograms
        f, t, Sxx = self.compute_db_spectrograms(wf)

        # Save f/t range once per station
        if station_tf is None:
            station_tf = np.array([float(f[0]), float(f[-1]), float(t[0]), float(t[-1])], dtype=float)
            np.save(st_out / "f_t_range.npy", station_tf)

        # Convert to RGB-like image: (F, T, 3)
        try:
            rgb = Sxx.transpose((1, 2, 0))
        except Exception as e:
            print(f"[SKIP] {st} {label_folder} {trace}: transpose failed shape={getattr(Sxx,'shape',None)} ({e})")
            return station_tf, None

        if rgb.shape[0] != self.expected_hw[0] or rgb.shape[1] != self.expected_hw[1]:
            print(f"[SKIP] {st} {label_folder} {trace}: unexpected HxW={rgb.shape[:2]} expected={self.expected_hw}")
            return station_tf, None

        filename = f"{st}__{trace}.png"
        out_path = st_out / split_name / label_folder / filename

        if not self._safe_save_rgb(out_path, rgb, st, label_folder, trace):
            return station_tf, None

        out = dict(row_src)
        out["station"] = st
        out["trace_name"] = str(trace)
        out["label"] = label_folder
        out["split"] = split_name
        out["filename"] = str(out_path.relative_to(st_out.parent.parent)).replace("\\", "/")  # relative to <OUT_ROOT>

        # If trace_start_time exists, compute ISO week
        if "trace_start_time" in out and pd.notna(out["trace_start_time"]):
            try:
                dt = pd.to_datetime(out["trace_start_time"])
                out["week"] = int(dt.isocalendar().week)
            except Exception:
                out["week"] = None
        else:
            out["week"] = None

        # Cleanup big arrays
        del wf, Sxx, rgb
        return station_tf, out

    def compute_db_spectrograms(self, waveform):
        specs = []
        f, t = None, None
        noverlap = int(self.noverlap)

        for ch in waveform:
            fi, ti, Sxx = scipy.signal.spectrogram(ch, fs=self.fs, nperseg=self.nperseg, noverlap=noverlap)
            specs.append(Sxx)
            if f is None:
                f, t = fi, ti

        specs = np.stack(specs, axis=0)    
        specs = np.log10(specs + self.eps) 

        if self.cut_freq is not None:
            try:
                cut_val = float(self.cut_freq)
                mask = f <= cut_val
                f = f[mask]
                specs = specs[:, mask, :]
            except Exception:
                pass

        if self.normalization:
            mn = specs.min()
            mx = specs.max()
            if mx - mn == 0:
                specs = specs * 0.0
            else:
                specs = (specs - mn) / (mx - mn)

        return f, t, specs

    def extract_waveform_features(self, waveform):
        features = {}
        for i in range(3):
            channel = waveform[i]
            channel_id = f"c{i}"

            features[f"f1_{channel_id}_maximum_amplitude"] = float(np.max(channel))
            features[f"f2_{channel_id}_minimum_amplitude"] = float(np.min(channel))
            features[f"f3_{channel_id}_mean_amplitude"] = float(np.mean(channel))
            features[f"f4_{channel_id}_std_dev_amplitude"] = float(np.std(channel))
            features[f"f5_{channel_id}_median_amplitude"] = float(np.median(channel))
            features[f"f6_{channel_id}_signal_range"] = float(np.ptp(channel))

            features[f"f7_{channel_id}_total_energy"] = float(np.sum(channel ** 2))
            features[f"f8_{channel_id}_root_mean_square"] = float(np.sqrt(np.mean(channel ** 2)))
            features[f"f9_{channel_id}_absolute_mean_amplitude"] = float(np.mean(np.abs(channel)))

            freqs, psd = scipy.signal.welch(channel)
            features[f"f10_{channel_id}_peak_spectral_power"] = float(np.max(psd))
            features[f"f11_{channel_id}_dominant_frequency"] = float(freqs[np.argmax(psd)])
            features[f"f12_{channel_id}_mean_spectral_power"] = float(np.mean(psd))

            psd_sum = np.sum(psd)
            if psd_sum == 0:
                features[f"f13_{channel_id}_spectral_entropy"] = 0.0
            else:
                features[f"f13_{channel_id}_spectral_entropy"] = float(scipy.stats.entropy(psd / psd_sum))

            psd_safe = np.where(psd <= 0, 1e-12, psd)
            features[f"f14_{channel_id}_spectral_flatness"] = float(
                np.exp(np.mean(np.log(psd_safe))) / np.mean(psd_safe)
            )

            zero_crosses = np.where(np.diff(np.signbit(channel)))[0]
            features[f"f15_{channel_id}_num_zero_crossings"] = int(len(zero_crosses))

            peaks = scipy.signal.find_peaks(channel)[0]
            features[f"f16_{channel_id}_num_peaks"] = int(len(peaks))

            features[f"f17_{channel_id}_kurtosis"] = float(scipy.stats.kurtosis(channel))
            features[f"f18_{channel_id}_skewness"] = float(scipy.stats.skew(channel))
            features[f"f19_{channel_id}_variance"] = float(np.var(channel))
            features[f"f20_{channel_id}_iqr"] = float(np.percentile(channel, 75) - np.percentile(channel, 25))

            peak_idx = int(np.argmax(channel))
            rise_time = peak_idx if peak_idx > 0 else 0
            fall_time = (len(channel) - peak_idx) if peak_idx < len(channel) else 0
            features[f"f21_{channel_id}_rise_time"] = int(rise_time)
            features[f"f22_{channel_id}_fall_time"] = int(fall_time)
            features[f"f23_{channel_id}_peak_to_peak_amplitude"] = float(np.max(channel) - np.min(channel))

        return features

    def _has_subfolders(self, folder):
        folder = Path(folder)
        return any(p.is_dir() for p in folder.iterdir())

    def _looks_like_station_folder(self, folder):
        """
        True only if the folder contains station files DIRECTLY inside it.
        """
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            return False

        # only count files in this folder, not in subfolders
        pre = list(folder.glob("*_pre.hdf5"))
        post = list(folder.glob("*_post.hdf5"))
        return len(pre) > 0 and len(post) > 0


        # Must contain at least one *_pre.hdf5 and one *_post.hdf5
        pre = list(folder.glob("*_pre.hdf5"))
        post = list(folder.glob("*_post.hdf5"))
        return len(pre) > 0 and len(post) > 0

    def _station_files(self, st_dir):

        st_dir = Path(st_dir)

        pre_candidates = sorted(st_dir.glob("*_pre.hdf5"))
        post_candidates = sorted(st_dir.glob("*_post.hdf5"))

        if not pre_candidates or not post_candidates:
            raise FileNotFoundError(f"Missing *_pre.hdf5 or *_post.hdf5 in {st_dir}")

        # Infer prefix from the first pre file: "<prefix>_pre.hdf5"
        pre_h5 = pre_candidates[0]
        prefix = pre_h5.name.replace("_pre.hdf5", "")

        # Prefer matching post file for same prefix if it exists
        post_h5 = st_dir / f"{prefix}_post.hdf5"
        if not post_h5.exists():
            # fallback to first post candidate
            post_h5 = post_candidates[0]

        pre_csv = self._pick_attributes_file(st_dir, prefix=f"{prefix}_pre_attribute")
        post_csv = self._pick_attributes_file(st_dir, prefix=f"{prefix}_post_attribute")

        return pre_h5, post_h5, pre_csv, post_csv

    def _pick_attributes_file(self, st_dir, prefix):

        st_dir = Path(st_dir)

        # exact candidates
        cand1 = st_dir / f"{prefix}.csv"
        cand2 = st_dir / f"{prefix}s.csv"

        if cand1.exists():
            return cand1
        if cand2.exists():
            return cand2

        # try without extension or any extension
        glob_any = sorted(st_dir.glob(prefix + "*"))
        # filter out hdf5
        glob_any = [p for p in glob_any if p.is_file() and p.suffix.lower() != ".hdf5"]
        if glob_any:
            return glob_any[0]

        raise FileNotFoundError(f"Attributes file not found in {st_dir} for prefix '{prefix}'")

    def _read_attributes(self, path):

        path = Path(path)

        # Try normal CSV read first
        try:
            df = pd.read_csv(path)
        except Exception:
            # If file has no extension or delimiter oddities, try common fallbacks
            df = pd.read_csv(path, sep=None, engine="python")

        if "trace_name" not in df.columns:
            raise ValueError(f"Attributes file missing 'trace_name' column: {path}")

        if "trace_start_time" in df.columns:
            try:
                df["trace_start_time"] = pd.to_datetime(df["trace_start_time"], errors="coerce")
            except Exception:
                pass

        return df

    def _split_traces(self, pre_traces, post_traces, train_frac, force_balance):
        pre_traces = list(map(str, pre_traces))
        post_traces = list(map(str, post_traces))

        random.shuffle(pre_traces)
        random.shuffle(post_traces)

        out = {"force_balance": bool(force_balance)}

        if force_balance:
            n = min(len(pre_traces), len(post_traces))
            pre_keep = pre_traces[:n]
            post_keep = post_traces[:n]

            n_train = int(np.floor(train_frac * n))

            out["train_pre"] = set(pre_keep[:n_train])
            out["test_pre"] = set(pre_keep[n_train:])
            out["train_post"] = set(post_keep[:n_train])
            out["test_post"] = set(post_keep[n_train:])
            out["used_pre"] = set(pre_keep)
            out["used_post"] = set(post_keep)
            out["n_balanced"] = int(n)
            out["n_train_per_class"] = int(n_train)
            return out

        # Not forced balanced: split each list separately
        n_pre = len(pre_traces)
        n_post = len(post_traces)

        n_train_pre = int(np.floor(train_frac * n_pre))
        n_train_post = int(np.floor(train_frac * n_post))

        out["train_pre"] = set(pre_traces[:n_train_pre])
        out["test_pre"] = set(pre_traces[n_train_pre:])
        out["train_post"] = set(post_traces[:n_train_post])
        out["test_post"] = set(post_traces[n_train_post:])

        out["used_pre"] = set(pre_traces)
        out["used_post"] = set(post_traces)
        out["n_balanced"] = None
        out["n_train_per_class"] = None
        return out

    def _build_index_by_trace(self, df):
        tmp = df.copy()
        tmp["trace_name"] = tmp["trace_name"].astype(str)
        tmp = tmp.drop_duplicates(subset=["trace_name"], keep="first")
        # Convert each row to a dict for quick lookup
        idx = {}
        for _, row in tmp.iterrows():
            idx[str(row["trace_name"])] = row.to_dict()
        return idx

    def _make_dirs(self, st_out):
        st_out = Path(st_out)
        for split in ("train", "test"):
            for cls in ("foreshock", "aftershock"):
                (st_out / split / cls).mkdir(parents=True, exist_ok=True)

    def _safe_save_rgb(self, path, rgb, st, label, trace):
        if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
            print(f"[SKIP] {st} {label} {trace}: rgb not 3D shape={getattr(rgb,'shape',None)}")
            return False

        if rgb.shape[2] not in (3, 4):
            print(f"[SKIP] {st} {label} {trace}: invalid channels={rgb.shape[2]} shape={rgb.shape}")
            return False

        if not np.isfinite(rgb).all():
            print(f"[SKIP] {st} {label} {trace}: NaN/Inf in rgb")
            return False

        # If float, clip to [0,1]
        if rgb.dtype.kind in ("f", "c"):
            rgb = np.clip(rgb, 0.0, 1.0)

        try:
            plt.imsave(str(path), rgb)
            return True
        except Exception as e:
            print(f"[SKIP] {st} {label} {trace}: imsave failed ({e}) shape={rgb.shape} dtype={rgb.dtype}")
            return False

    def _fix_length(self, wf, n_samples):
        wf = wf[:, :n_samples]
        if wf.shape[1] < n_samples:
            pad = n_samples - wf.shape[1]
            wf = np.pad(wf, ((0, 0), (0, pad)), mode="constant")
        return wf

    def _expected_hw(self, n_samples):
        hop = self.nperseg - self.noverlap
        if hop <= 0:
            raise ValueError(f"Invalid hop={hop}. Check nperseg/noverlap_ratio.")
        F_bins = self.nperseg // 2 + 1
        T_bins = 1 + int(np.floor((n_samples - self.nperseg) / hop))
        return (F_bins, T_bins)