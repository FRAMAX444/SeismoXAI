import os
import gc
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from torchvision import transforms
from torchmetrics.functional import confusion_matrix

import matplotlib.pyplot as plt


class CNN2D(pl.LightningModule):
    def __init__(self, input_dim, n_classes=2, lr=1e-3, dropout=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["input_dim"])  # keep checkpoint clean
        self.lr = float(lr)
        self.n_classes = int(n_classes)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=float(dropout))
        self.flatten = nn.Flatten()

        dummy = torch.zeros((1, 3, int(input_dim[0]), int(input_dim[1])))
        flat = self._get_flattened_size(dummy)

        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, self.n_classes)

        self.criterion = nn.CrossEntropyLoss()

        # will be created on test start
        self.total_conf_matrix = None

    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.relu(self.maxpool(self.bn1(self.conv1(x))))
            x = self.relu(self.maxpool(self.bn2(self.conv2(x))))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.flatten(x)
        return x.shape[1]

    def forward(self, x):
        x = self.relu(self.maxpool(self.bn1(self.conv1(x))))
        x = self.relu(self.maxpool(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def on_test_epoch_start(self):
        self.total_conf_matrix = None

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)

        if self.total_conf_matrix is None:
            self.total_conf_matrix = torch.zeros(self.n_classes, self.n_classes, device=x.device)

        cm = confusion_matrix(preds, y, task="multiclass", num_classes=self.n_classes)
        self.total_conf_matrix += cm

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

class FolderSpectrogramDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = [Path(p) for p in files]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        label_folder = img_path.parent.name.lower()

        if label_folder == "foreshock":
            y = 0
        elif label_folder == "aftershock":
            y = 1
        else:
            raise ValueError(f"Cannot infer label from path: {img_path}")

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            x = self.transform(im) if self.transform is not None else transforms.ToTensor()(im)

        return x, torch.tensor(y, dtype=torch.long)


class Training:
    def __init__(
        self,
        lr=1e-3,
        dropout=0.1,
        batch_size=32,
        max_epochs=30,
        patience=5,
        num_workers=0,
        max_images_for_stats=None,
        reproducible=True,
        use_cuda=True,
        seed=42,
        verbose=True
    ):
        self.lr = float(lr)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.num_workers = int(num_workers)
        self.max_images_for_stats = max_images_for_stats  # None or int
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.on_cuda = bool(use_cuda) and torch.cuda.is_available()
        self.reproducible = bool(reproducible)
        self.pin_memory = False

        if self.on_cuda:
            self.pin_memory = torch.cuda.is_available()

        if self.reproducible:
            print("[INFO] Training set to REPRODUCIBLE mode (deterministic).")
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            pl.seed_everything(self.seed, workers=True)
            os.environ["PYTHONHASHSEED"] = str(self.seed)

    def train(self, input_dir, output_dir, save_models=True, save_results=True):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        models_root = output_dir / "models"
        results_root = output_dir / "results"
        if save_models:
            models_root.mkdir(parents=True, exist_ok=True)
        if save_results:
            results_root.mkdir(parents=True, exist_ok=True)

        station_dirs = self._discover_station_dirs(input_dir)
        if not station_dirs:
            raise FileNotFoundError(
                "No station folders found. Expected either:\n"
                "- a station folder containing train/ and test/\n"
                "- a root folder containing station folders"
            )

        if self.verbose:
            print("Preprocessed input:", input_dir)
            print("Output dir:", output_dir)
            print("Stations discovered:", [p.name for p in station_dirs])
            print("save_models =", save_models, "| save_results =", save_results)

        for st_dir in station_dirs:
            try:
                self._train_one_station(
                    station_dir=st_dir,
                    models_root=models_root,
                    results_root=results_root,
                    save_models=save_models,
                    save_results=save_results
                )
            except Exception as e:
                print(f"[ERROR] {st_dir.name} failed: {e}")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def _discover_station_dirs(self, preprocessed_dir):
        if self._looks_like_station_folder(preprocessed_dir):
            return [preprocessed_dir]

        out = []
        if preprocessed_dir.exists():
            for p in sorted(preprocessed_dir.iterdir()):
                if p.is_dir() and self._looks_like_station_folder(p):
                    out.append(p)
        return out

    def _looks_like_station_folder(self, d):
        d = Path(d)
        return (
            d.is_dir()
            and (d / "train").exists()
            and (d / "test").exists()
            and (d / "input_dim.npy").exists()
            and (d / "train" / "foreshock").exists()
            and (d / "train" / "aftershock").exists()
            and (d / "test" / "foreshock").exists()
            and (d / "test" / "aftershock").exists()
        )

    def _train_one_station(self, station_dir, models_root, results_root, save_models, save_results):
        station_dir = Path(station_dir)
        station = station_dir.name

        h, w = np.load(station_dir / "input_dim.npy").astype(int).tolist()
        input_dim = (int(h), int(w))

        train_files = self._list_pngs(station_dir / "train")
        test_files = self._list_pngs(station_dir / "test")

        if len(train_files) == 0 or len(test_files) == 0:
            print(f"[SKIP] {station}: train_files={len(train_files)} test_files={len(test_files)}")
            return

        mean, std = self._compute_streaming_mean_std(train_files, max_images=self.max_images_for_stats)

        model_out_dir = None
        results_out_dir = None

        if save_models:
            model_out_dir = models_root / station
            model_out_dir.mkdir(parents=True, exist_ok=True)

            (model_out_dir / "stats.json").write_text(
                json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
            )
            (model_out_dir / "run_config.json").write_text(json.dumps({
                "station": station,
                "input_dim": [input_dim[0], input_dim[1]],
                "lr": self.lr,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "num_workers": self.num_workers,
                "max_images_for_stats": self.max_images_for_stats,
                "train_files": len(train_files),
                "test_files": len(test_files),
            }, indent=2))

        if save_results:
            results_out_dir = results_root / station
            results_out_dir.mkdir(parents=True, exist_ok=True)

        tfm = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])

        train_ds = FolderSpectrogramDataset(train_files, transform=tfm)
        test_ds = FolderSpectrogramDataset(test_files, transform=tfm)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

        model = CNN2D(input_dim=input_dim, n_classes=2, lr=self.lr, dropout=self.dropout)

        callbacks = []
        logger = None
        ckpt = None

        if save_models:
            ckpt = ModelCheckpoint(
                dirpath=str(model_out_dir),
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True
            )
            callbacks.append(ckpt)
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=self.patience, verbose=True))
            logger = CSVLogger(save_dir=str(model_out_dir), name="logs")

        if self.verbose:
            print(f"\n===== TRAIN {station} | train_files={len(train_files)} test_files={len(test_files)} | input_dim={input_dim} =====")

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices="auto",
            deterministic=True,
            log_every_n_steps=10,
            callbacks=callbacks if callbacks else None,
            logger=logger
        )

        trainer.fit(model, train_loader, test_loader)

        best_ckpt_path = None
        eval_model = model

        if save_models and ckpt is not None and ckpt.best_model_path:
            best_ckpt_path = ckpt.best_model_path
            eval_model = CNN2D.load_from_checkpoint(
                best_ckpt_path,
                input_dim=input_dim,
                n_classes=2,
                lr=self.lr,
                dropout=self.dropout
            )

        conf_torch = self._run_test_and_get_confusion(eval_model, test_loader)

        self._print_confusion_matrix(station, conf_torch)

        if save_results:
            cm_path = results_out_dir / "confusion_matrix.png"
            self._save_confusion_matrix_png(station, conf_torch, cm_path)

            metrics = self._metrics_from_confusion(conf_torch)
            metrics_path = results_out_dir / "metrics.json"
            payload = {
                "station": station,
                "best_checkpoint": best_ckpt_path,
                "input_dim": [input_dim[0], input_dim[1]],
                "test_files": len(test_files),
                "confusion_matrix": conf_torch.tolist(),
                "metrics": metrics,
            }
            metrics_path.write_text(json.dumps(payload, indent=2))

            if self.verbose:
                print(f"[OK] results saved: {cm_path} and {metrics_path}")

        if save_models and self.verbose:
            print(f"[DONE] {station} model saved in {model_out_dir} (best={best_ckpt_path})")

        del trainer, model, eval_model, train_loader, test_loader, train_ds, test_ds, train_files, test_files
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_test_and_get_confusion(self, model, dataloader):
        model.total_conf_matrix = None

        test_trainer = pl.Trainer(
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_checkpointing=False,
            deterministic=True
        )
        test_trainer.test(model=model, dataloaders=dataloader, verbose=False)

        if model.total_conf_matrix is None:
            raise RuntimeError("Confusion matrix was not accumulated (total_conf_matrix is None).")

        cm = model.total_conf_matrix.detach().cpu().to(torch.int64).numpy()
        return cm

    def _print_confusion_matrix(self, station, conf_2x2):
        TN, FP, FN, TP = conf_2x2.flatten().tolist()

        print("\nConfusion Matrix (TEST SET) —", station)
        print("=======================================")
        print("               Pred Foreshock  Pred Aftershock")
        print(f"Actual Foreshock      {TN:6d}         {FP:6d}")
        print(f"Actual Aftershock     {FN:6d}         {TP:6d}")
        print("=======================================")

        metrics = self._metrics_from_confusion(conf_2x2)
        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall   : {metrics['recall']:.4f}")
        print(f"F1-score : {metrics['f1_score']:.4f}\n")

    def _metrics_from_confusion(self, conf_2x2):
        TN, FP, FN, TP = conf_2x2.flatten().tolist()
        total = TN + FP + FN + TP

        acc = (TP + TN) / total if total else 0.0
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "TP": int(TP),
            "support": int(total),
        }

    def _save_confusion_matrix_png(self, station, conf_2x2, out_path):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        im = ax.imshow(conf_2x2, cmap="Blues", vmin=0)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Foreshock", "Aftershock"])
        ax.set_yticklabels(["Foreshock", "Aftershock"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {station}")

        max_val = conf_2x2.max() if conf_2x2.max() > 0 else 1
        for (i, j), v in np.ndenumerate(conf_2x2):
            color = "white" if v > 0.6 * max_val else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    def _list_pngs(self, split_dir):
        split_dir = Path(split_dir)
        files = []
        for cls in ("foreshock", "aftershock"):
            folder = split_dir / cls
            if not folder.exists():
                continue
            files.extend(sorted(folder.glob("*.png")))
        return files

    def _compute_streaming_mean_std(self, files, max_images=None):
        files = list(files)
        if max_images is not None:
            files = files[: int(max_images)]

        sum_c = np.zeros(3, dtype=np.float64)
        sumsq_c = np.zeros(3, dtype=np.float64)
        count = 0

        for p in tqdm(files, desc="Compute mean/std", disable=not self.verbose):
            p = Path(p)
            if not p.exists():
                continue

            with Image.open(p) as im:
                im = im.convert("RGB")
                arr = np.asarray(im, dtype=np.float32) / 255.0

            if arr.ndim != 3 or arr.shape[2] != 3:
                continue

            pixels = arr.reshape(-1, 3)
            sum_c += pixels.sum(axis=0)
            sumsq_c += (pixels ** 2).sum(axis=0)
            count += pixels.shape[0]

            del arr, pixels

        if count == 0:
            raise ValueError("No pixels counted for mean/std (missing images?)")

        mean = sum_c / count
        var = (sumsq_c / count) - (mean ** 2)
        var = np.maximum(var, 1e-12)
        std = np.sqrt(var)

        return mean.astype(np.float32), std.astype(np.float32)

    def train_one_model_multi_stations(
        self,
        input_dir,
        output_dir,
        stations=None,  # None => all; otherwise list of station folder names
        model_name=None,
        save_models=True,
        save_results=True,
    ):
        """
        Traina UNA sola CNN su un sottoinsieme di stazioni.
        - stations=None => tutte le stazioni
        - stations=['AAA','BBB'] => solo quelle stazioni (nomi cartelle)
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        station_dirs = self._discover_station_dirs(input_dir)
        if not station_dirs:
            raise FileNotFoundError(
                "No station folders found. Expected station folders containing train/ and test/."
            )

        # -------------------------
        # Filter stations if provided
        # -------------------------
        all_by_name = {p.name: p for p in station_dirs}

        if stations is None:
            selected_dirs = station_dirs
            selected_names = [p.name for p in station_dirs]
        else:
            if isinstance(stations, str):
                stations = [stations]
            requested = list(stations)
            missing = [s for s in requested if s not in all_by_name]
            if missing:
                available = sorted(all_by_name.keys())
                raise ValueError(
                    "Some requested stations were not found:\n"
                    f"  missing: {missing}\n"
                    f"  available: {available}"
                )
            selected_dirs = [all_by_name[s] for s in requested]
            selected_names = requested

        if model_name is None:
            model_name = "MODEL_" + "_".join(selected_names)

        # -------------------------
        # Check train/test folders and input_dim consistency
        # -------------------------
        dims = {}
        for st_dir in selected_dirs:
            st = st_dir.name

            if not (st_dir / "train").exists() or not (st_dir / "test").exists():
                raise FileNotFoundError(
                    f"Station '{st}' is missing 'train/' or 'test/' folders: {st_dir}"
                )

            dim_path = st_dir / "input_dim.npy"
            if not dim_path.exists():
                raise FileNotFoundError(f"Missing input_dim.npy for station '{st}': {dim_path}")

            h, w = np.load(dim_path).astype(int).tolist()
            dims[st] = (int(h), int(w))

        uniq_dims = sorted(set(dims.values()))
        if len(uniq_dims) != 1:
            msg = "Input dimensions are NOT consistent across selected stations:\n"
            for st, d in sorted(dims.items()):
                msg += f"  - {st}: {d}\n"
            msg += f"Unique dims found: {uniq_dims}\n"
            raise ValueError(msg)

        input_dim = uniq_dims[0]

        # -------------------------
        # Collect global train/test files for selected stations
        # -------------------------
        all_train_files, all_test_files = [], []
        for st_dir in selected_dirs:
            all_train_files.extend(self._list_pngs(st_dir / "train"))
            all_test_files.extend(self._list_pngs(st_dir / "test"))

        if len(all_train_files) == 0 or len(all_test_files) == 0:
            raise ValueError(
                f"Empty dataset for selected stations: train_files={len(all_train_files)} test_files={len(all_test_files)}"
            )

        # -------------------------
        # Compute global mean/std on TRAIN ONLY (selected stations)
        # -------------------------
        mean, std = self._compute_streaming_mean_std(
            all_train_files, max_images=self.max_images_for_stats
        )

        # -------------------------
        # Output folders
        # -------------------------
        models_root = output_dir / "models"
        results_root = output_dir / "results"
        if save_models:
            models_root.mkdir(parents=True, exist_ok=True)
        if save_results:
            results_root.mkdir(parents=True, exist_ok=True)

        model_out_dir = None
        results_out_dir = None

        if save_models:
            model_out_dir = models_root / model_name
            model_out_dir.mkdir(parents=True, exist_ok=True)

            (model_out_dir / "stats.json").write_text(
                json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2)
            )
            (model_out_dir / "run_config.json").write_text(json.dumps({
                "model_name": model_name,
                "stations": selected_names,
                "input_dim": [input_dim[0], input_dim[1]],
                "lr": self.lr,
                "dropout": self.dropout,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "num_workers": self.num_workers,
                "max_images_for_stats": self.max_images_for_stats,
                "train_files": len(all_train_files),
                "test_files": len(all_test_files),
            }, indent=2))

        if save_results:
            results_out_dir = results_root / model_name
            results_out_dir.mkdir(parents=True, exist_ok=True)

        # -------------------------
        # Dataloaders
        # -------------------------
        tfm = transforms.Compose([
            transforms.Resize(input_dim),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
        ])

        train_ds = FolderSpectrogramDataset(all_train_files, transform=tfm)
        test_ds = FolderSpectrogramDataset(all_test_files, transform=tfm)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

        # -------------------------
        # Model + Trainer
        # -------------------------
        model = CNN2D(input_dim=input_dim, n_classes=2, lr=self.lr, dropout=self.dropout)

        callbacks = []
        logger = None
        ckpt = None

        if save_models:
            ckpt = ModelCheckpoint(
                dirpath=str(model_out_dir),
                filename="best",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_last=True
            )
            callbacks.append(ckpt)
            callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=self.patience, verbose=True))
            logger = CSVLogger(save_dir=str(model_out_dir), name="logs")

        if self.verbose:
            print("\n===== TRAIN ONE MODEL ON SELECTED STATIONS =====")
            print("Stations:", selected_names)
            print(f"Global train_files={len(all_train_files)} | test_files={len(all_test_files)} | input_dim={input_dim}")

        trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator="auto",
            devices="auto",
            deterministic=True,
            log_every_n_steps=10,
            callbacks=callbacks if callbacks else None,
            logger=logger
        )

        trainer.fit(model, train_loader, test_loader)

        # -------------------------
        # Evaluate best checkpoint (if available) else last weights
        # -------------------------
        best_ckpt_path = None
        eval_model = model

        if save_models and ckpt is not None and ckpt.best_model_path:
            best_ckpt_path = ckpt.best_model_path
            eval_model = CNN2D.load_from_checkpoint(
                best_ckpt_path,
                input_dim=input_dim,
                n_classes=2,
                lr=self.lr,
                dropout=self.dropout
            )

        conf_torch = self._run_test_and_get_confusion(eval_model, test_loader)
        self._print_confusion_matrix(model_name, conf_torch)

        if save_results:
            cm_path = results_out_dir / "confusion_matrix.png"
            self._save_confusion_matrix_png(model_name, conf_torch, cm_path)

            metrics = self._metrics_from_confusion(conf_torch)
            metrics_path = results_out_dir / "metrics.json"
            payload = {
                "model_name": model_name,
                "stations": selected_names,
                "best_checkpoint": best_ckpt_path,
                "input_dim": [input_dim[0], input_dim[1]],
                "test_files": len(all_test_files),
                "confusion_matrix": conf_torch.tolist(),
                "metrics": metrics,
            }
            metrics_path.write_text(json.dumps(payload, indent=2))

            if self.verbose:
                print(f"[OK] results saved: {cm_path} and {metrics_path}")

        if save_models and self.verbose:
            print(f"[DONE] model saved in {model_out_dir} (best={best_ckpt_path})")

        # Cleanup
        del trainer, model, eval_model, train_loader, test_loader, train_ds, test_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
