"""
Generate tile-level predictions from the final EfficientNet-B3 model.

The output CSV files are used for spatial reconstruction of ground-truth labels,
model predictions, and classification errors on the original thin-section images.
"""


# %%
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.efficientnet import preprocess_input

# =========================
# Configuration
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_ROOT = os.path.join(REPO_DIR, "data", "DATASET_FINAL")
CSV_PATH = os.path.join(REPO_DIR, "metadata", "dataset_labels_esin_split.csv")
RESULT_DIR = os.path.join(REPO_DIR, "results")

RUN_NAME = "efficientnetb3_finetune_v6a_onestop"

BEST_VALACC_PATH = os.path.join(
    REPO_DIR,
    "weights",
    "efficientnetb3_final_model.keras"
)

IMG_SIZE = (180, 180)
NUM_CLASSES = 10
BATCH_SIZE = 64

CLASS_NAMES = [
    "intraclast",
    "Girvanella crust",
    "thromboid",
    "microbial peloid",
    "peloid",
    "coated grain",
    "bioclast",
    "quartz",
    "micrite",
    "cement",
]

def out_path(suffix: str) -> str:
    return os.path.join(RESULT_DIR, f"{RUN_NAME}_{suffix}")

if not os.path.exists(BEST_VALACC_PATH):
    raise FileNotFoundError(f"Model file not found: {BEST_VALACC_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

print(f"Using model: {BEST_VALACC_PATH}")
print(f"Using dataset CSV: {CSV_PATH}")

# =========================
# Load metadata
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = {"filepath", "label", "subset", "sample"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing columns in dataset_labels.csv: {missing_cols}")

test_mf1_df = df[df["subset"] == "test_mf1"].copy().reset_index(drop=True)
test_mf3_df = df[df["subset"] == "test_mf3"].copy().reset_index(drop=True)

def build_paths_labels(sub_df):
    rel_paths = sub_df["filepath"].astype(str).to_numpy()
    full_paths = np.array([os.path.join(DATA_ROOT, p) for p in rel_paths])
    labels = sub_df["label"].astype(int).to_numpy()
    samples = sub_df["sample"].astype(str).to_numpy()
    return full_paths, labels, rel_paths, samples

X_test_mf1, y_test_mf1, rel_mf1, sample_mf1 = build_paths_labels(test_mf1_df)
X_test_mf3, y_test_mf3, rel_mf3, sample_mf3 = build_paths_labels(test_mf3_df)

print(f"MF1 test: {len(X_test_mf1)}")
print(f"MF3 test: {len(X_test_mf3)}")

# Check image file availability
missing_files = [p for p in np.concatenate([X_test_mf1, X_test_mf3]) if not os.path.exists(p)]
if missing_files:
    print(f"[Warning] Missing image files: {len(missing_files)}")
    print("First missing file:", missing_files[0])
else:
    print("All test image files found.")

# =========================
# Extract tile index
# =========================
def extract_tile_index(filepath):
    """
    Example:
    images/test_mf1/(cali.) 20150212 HXND 2-3 (1)_06122.png
    -> 6122
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    tile_str = base.rsplit("_", 1)[-1]
    return int(tile_str)

def extract_filename(filepath):
    return os.path.basename(filepath)

# =========================
# Build tf.data pipelines
# =========================
def preprocess_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def make_dataset(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

mf1_ds = make_dataset(X_test_mf1, y_test_mf1)
mf3_ds = make_dataset(X_test_mf3, y_test_mf3)

# =========================
# Load model
# =========================
model = tf.keras.models.load_model(BEST_VALACC_PATH, compile=False)
print("Loaded best_valacc model.")

# =========================
# Prediction and output writer
# =========================
def save_all_predictions(
    model,
    ds,
    y_true,
    rel_paths,
    full_paths,
    samples,
    subset_name
):
    probs = model.predict(ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    tile_indices = [extract_tile_index(p) for p in rel_paths]

    out_df = pd.DataFrame({
        "filepath": rel_paths,
        "full_path": full_paths,
        "sample": samples,
        "filename": [extract_filename(p) for p in rel_paths],
        "tile_index": tile_indices,
        "true_label": y_true.astype(int),
        "pred_label": y_pred.astype(int),
        "true_name": [CLASS_NAMES[i] for i in y_true],
        "pred_name": [CLASS_NAMES[i] for i in y_pred],
        "pred_confidence": confs,
        "correct": y_true.astype(int) == y_pred.astype(int),
    })

    for i in range(NUM_CLASSES):
        out_df[f"prob_class_{i}"] = probs[:, i]

    save_path = out_path(f"best_valacc_{subset_name}_all_predictions.csv")
    out_df.to_csv(save_path, index=False)

    acc = out_df["correct"].mean()
    print(f"\nSaved: {save_path}")
    print(f"{subset_name.upper()} accuracy from all_predictions: {acc:.4f}")
    print(f"Total rows: {len(out_df)}")

    return out_df

mf1_pred_df = save_all_predictions(
    model=model,
    ds=mf1_ds,
    y_true=y_test_mf1,
    rel_paths=rel_mf1,
    full_paths=X_test_mf1,
    samples=sample_mf1,
    subset_name="mf1"
)

mf3_pred_df = save_all_predictions(
    model=model,
    ds=mf3_ds,
    y_true=y_test_mf3,
    rel_paths=rel_mf3,
    full_paths=X_test_mf3,
    samples=sample_mf3,
    subset_name="mf3"
)

print("\nDone.")
