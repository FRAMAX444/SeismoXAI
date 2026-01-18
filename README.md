![SeismoXAI Header](media/header.png)

End‑to‑end pipeline for **foreshock vs aftershock classification** from seismic waveforms using CNNs, with **SHAP‑based explainability** on time–frequency representations.

The repository is organized in three main stages:

1. **Preprocessing** – convert raw waveform data (HDF5) into spectrogram images
2. **Training** – train a CNN classifier per station
3. **Explainability (SHAP)** – compute and visualize SHAP attribution maps on correctly classified samples

---

## Repository Structure

```
.
├── preprocessing.py        # Dataset generation (spectrograms + metadata)
├── training.py             # CNN model + training pipeline
├── shap_explainer.py       # SHAP explainability pipeline
├── requirements.txt        # Python dependencies
├── preprocessed_dset*/     # Generated datasets (one folder per station)
├── trained_models/         # Trained models and results
│   ├── models/
│   │   └── <STATION>/
│   └── results/
└── trained_models_correct_test_samples/
```

Each **station** is processed, trained, and explained independently.

---

## Installation

Create a virtual environment (recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Data Preparation (Preprocessing)

### Expected Input Format

Each station folder must contain:

```
<STATION>/
├── <STATION>_pre.hdf5          # foreshock waveforms
├── <STATION>_post.hdf5         # aftershock waveforms
├── <STATION>_pre_attribute.csv
└── <STATION>_post_attribute.csv
```

Waveforms must be shaped as `(3, N)` (3 components).

You can pass:
- a **single station folder**, or
- a **root folder containing multiple station folders**

---

### Running Preprocessing

```python
from preprocessing import Preprocessing

pp = Preprocessing(
    n_samples=1300,
    fs=100,
    nperseg=64,
    noverlap_ratio=0.8,
    normalization=True,
    train_frac=0.8,
    force_balance=True
)

pp.make_dset(
    input_dir="raw_data",
    output_dir="preprocessed_dset"
)
```

### Output

For each station:

```
<STATION>/
├── train/
│   ├── foreshock/*.png
│   └── aftershock/*.png
├── test/
│   ├── foreshock/*.png
│   └── aftershock/*.png
├── input_dim.npy
├── input_dim.json
├── f_t_range.npy
└── metadata.csv
```

Spectrograms are saved as **RGB images** (frequency × time × channel).

---

## 2. Model Training

Each station is trained independently with the same CNN architecture.

### Running Training

```python
from training import Training

trainer = Training(
    lr=1e-3,
    batch_size=32,
    max_epochs=30,
    patience=5,
    reproducible=True
)

trainer.train(
    input_dir="preprocessed_dset",
    output_dir="trained_models",
    save_models=True,
    save_results=True
)
```

### Output

For each station:

```
trained_models/
├── models/<STATION>/
│   ├── best.ckpt
│   ├── stats.json
│   ├── run_config.json
│   └── logs/
└── results/<STATION>/
    ├── confusion_matrix.png
    └── metrics.json
```

The model is a 2D CNN trained on spectrogram images with normalization statistics stored per station.

---

## 3. SHAP Explainability

The SHAP pipeline:

1. Selects **correctly classified test samples**
2. Computes SHAP attribution maps per sample
3. Averages SHAP maps per class
4. Provides multiple plotting utilities

### Initialize the Explainer

```python
from shap_explainer import SHAPExplainer

explainer = SHAPExplainer(
    trained_models_dir="trained_models/models",
    preprocessed_dir="preprocessed_dset",
    correct_samples_dir="trained_models_correct_test_samples",
    verbose=True
)
```

### Compute SHAP for All Stations

```python
explainer.compute_all(
    max_per_class=50,
    max_evals=500,
    overwrite_shap=False,
    overwrite_means=True
)
```

This will automatically:
- discover stations
- extract correctly classified samples
- compute SHAP tensors
- compute mean SHAP maps

---

## Visualization Examples

### Single Station, One Class

```python
explainer.plot_station_single(
    station="ST01",
    which="pre"   # or "post"
)
```

### Single Station, Pre vs Post

```python
explainer.plot_station_both("ST01")
```

### Multiple Stations, Same Class

```python
explainer.plot_stations_single(
    stations=["ST01", "ST02", "ST03"],
    which="pre"
)
```

### Multiple Stations, Pre vs Post

```python
explainer.plot_stations_both(["ST01", "ST02", "ST03"])
```

SHAP maps are plotted in **frequency–time space** using the original axes from preprocessing.

---

## Notes & Best Practices

- Use **balanced datasets** (`force_balance=True`) for stable SHAP results
- SHAP Image maskers require **recent SHAP versions**
- If `shap.maskers.Image` is unavailable, the code falls back to `GradientExplainer`
- GPU is supported but optional
- Each station is fully independent → easy parallelization