"""
Train the final EfficientNet-B3 classification model used in the ESIN study.

This script implements the selected G6a configuration: EfficientNet-B3 with
ImageNet pretraining, 180 x 180 input resolution, cosine learning-rate decay,
partial fine-tuning of the upper backbone layers, label smoothing, class-1
undersampling, and minority-class oversampling.
"""


# %%
import os
import time
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support
)

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy


# %%
# =========================
# 0. Configuration
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_ROOT = os.path.join(REPO_DIR, "data", "DATASET_FINAL")
CSV_PATH  = os.path.join(REPO_DIR, "metadata", "dataset_labels_esin_split.csv")


IMG_SIZE    = (180, 180)  # 144 to 180 px input resolution.
NUM_CLASSES = 10
BATCH_SIZE  = 64
EPOCHS      = 100         # 150 to 100 epochs after validation plateau.

RUN_NAME   = "efficientnetb3_finetune_v6a_onestop"
RESULT_DIR = os.path.join(REPO_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]

OVERSAMPLE_MIN_COUNT   = 300
CLS1_UNDERSAMPLE_COUNT = 3000
LABEL_SMOOTHING        = 0.10
DROPOUT_RATE           = 0.45

UNFREEZE_LAST_N = 100
FINETUNE_LR     = 5e-6   # same as G4a/G5c
COSINE_ALPHA    = 0.02   # final LR = 5e-6 x 0.02 = 1e-7

def out_path(suffix: str) -> str:
    return os.path.join(RESULT_DIR, f"{RUN_NAME}_{suffix}")

print(f"RUN_NAME:       {RUN_NAME}")
print(f"BACKBONE:       EfficientNetB3 (ImageNet pretrained)")
print(f"IMG_SIZE:       {IMG_SIZE}  (G5b 180px)")
print(f"BATCH_SIZE:     {BATCH_SIZE}")
print(f"EPOCHS:         {EPOCHS}  (150→100)")
print(f"FINETUNE_LR:    {FINETUNE_LR}")
print(f"SCHEDULER:      CosineDecay (alpha={COSINE_ALPHA}, final={FINETUNE_LR*COSINE_ALPHA:.0e})")
print(f"UNFREEZE_LAST:  {UNFREEZE_LAST_N}")
print(f"DROPOUT:        {DROPOUT_RATE}")
print(f"LABEL_SMOOTH:   {LABEL_SMOOTHING}")
print(f"CLS1_US:        {CLS1_UNDERSAMPLE_COUNT}")


# %%
# =========================
# 1. Load metadata and prepare datasets
# =========================
df = pd.read_csv(CSV_PATH)

train_df    = df[df["subset"] == "train"].copy()
val_df      = df[df["subset"] == "val"].copy()
test_mf1_df = df[df["subset"] == "test_mf1"].copy()
test_mf3_df = df[df["subset"] == "test_mf3"].copy()

print("\n[Before Sampling] train class counts")
print(train_df["label"].value_counts().sort_index())

def prepare_train_df(train_df, min_count=300, cls1_undersample=3000, seed=42):
    result_parts = []
    for class_id in range(NUM_CLASSES):
        cls_df = train_df[train_df["label"].astype(int) == class_id].copy()
        n = len(cls_df)
        if n == 0:
            print(f"[Warning] class {class_id} has 0 samples. Skipping.")
            continue
        if class_id == 1 and n > cls1_undersample:
            cls_df = cls_df.sample(n=cls1_undersample, replace=False, random_state=seed)
            print(f"class {class_id}: {n} -> {len(cls_df)} (undersampled)")
        elif len(cls_df) < min_count:
            add_n = min_count - len(cls_df)
            sampled_df = cls_df.sample(n=add_n, replace=True, random_state=seed)
            cls_df = pd.concat([cls_df, sampled_df], axis=0, ignore_index=True)
            print(f"class {class_id}: {n} -> {len(cls_df)} (oversampled +{add_n})")
        else:
            print(f"class {class_id}: {n} -> {len(cls_df)} (kept)")
        result_parts.append(cls_df)

    out_df = pd.concat(result_parts, axis=0, ignore_index=True)
    out_df = out_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out_df

train_df = prepare_train_df(
    train_df=train_df,
    min_count=OVERSAMPLE_MIN_COUNT,
    cls1_undersample=CLS1_UNDERSAMPLE_COUNT,
    seed=SEED
)

print("\n[After Sampling] train class counts")
print(train_df["label"].value_counts().sort_index())

def build_paths_labels(sub_df):
    paths  = [os.path.join(DATA_ROOT, p) for p in sub_df["filepath"].tolist()]
    labels = sub_df["label"].astype(int).to_numpy()
    return np.array(paths), labels

X_train,    y_train     = build_paths_labels(train_df)
X_val,      y_val       = build_paths_labels(val_df)
X_test_mf1, y_test_mf1 = build_paths_labels(test_mf1_df)
X_test_mf3, y_test_mf3 = build_paths_labels(test_mf3_df)

print(f"\nTrain:{len(X_train)}, Val:{len(X_val)}, MF1:{len(X_test_mf1)}, MF3:{len(X_test_mf3)}")

pd.DataFrame({
    "class_id":     np.arange(NUM_CLASSES),
    "before_count": [int(df[df["subset"]=="train"]["label"].value_counts().get(i, 0))
                     for i in range(NUM_CLASSES)],
    "after_count":  [int(train_df["label"].value_counts().get(i, 0))
                     for i in range(NUM_CLASSES)],
}).to_csv(out_path("sampling_log.csv"), index=False)


# %%
# =========================
# 2. Build tf.data pipelines
# =========================
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomFlip("vertical"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.05),
])

def preprocess_img(path, label, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    if training:
        img = data_augmentation(img)
    img = preprocess_input(img)
    label = tf.one_hot(label, NUM_CLASSES)
    return img, label

def make_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda x, y: preprocess_img(x, y, training),
                num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(min(len(paths), 10000), seed=SEED,
                        reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_train,    y_train,    training=True)
val_ds   = make_dataset(X_val,      y_val,      training=False)
mf1_ds   = make_dataset(X_test_mf1, y_test_mf1, training=False)
mf3_ds   = make_dataset(X_test_mf3, y_test_mf3, training=False)


# %%
# =========================
# 3. Build EfficientNet-B3 and configure fine-tuning
# =========================
print("\n[Building EfficientNet-B3]")

base_model = EfficientNetB3(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMG_SIZE, 3)
)
print(f"Backbone: {base_model.name}")
print(f"Backbone total layers: {len(base_model.layers)}")
print(f"Backbone params: {base_model.count_params():,}")

# Classification head
x      = base_model.output
x      = GlobalAveragePooling2D()(x)
x      = Dense(512, activation="relu")(x)
x      = Dropout(DROPOUT_RATE)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model  = Model(inputs=base_model.input, outputs=output)

# Freeze / unfreeze configuration
# 1. Freeze the full backbone
for layer in base_model.layers:
    layer.trainable = False

# 2. Unfreeze the upper backbone layers
for layer in base_model.layers[-UNFREEZE_LAST_N:]:
    layer.trainable = True

# 3. Freeze all batch-normalization layers at the full-model level
#    This includes batch-normalization layers outside base_model.layers.
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# 4. Do not explicitly override trainability of the newly added head.
#    Newly added Dense and Dropout layers are trainable by default.
#    Overriding trainability here may undo the batch-normalization freeze.

# Configuration check
total_params     = model.count_params()
trainable_params = sum(tf.size(w).numpy() for w in model.trainable_weights)
frozen_bn        = sum(1 for l in model.layers
                       if isinstance(l, tf.keras.layers.BatchNormalization)
                       and not l.trainable)

print(f"\n[Fine-tuning configuration]")
print(f"  Backbone total layers: {len(base_model.layers)}")
print(f"  Unfreeze top-{UNFREEZE_LAST_N} ({UNFREEZE_LAST_N/len(base_model.layers)*100:.1f}%)")
print(f"  BN frozen: {frozen_bn} layers (full model scope)")
print(f"  Total params:     {total_params:,}")
print(f"  Trainable params: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
print(f"  LR: {FINETUNE_LR}")
print(f"  Scheduler: CosineDecay (alpha={COSINE_ALPHA}, final={FINETUNE_LR*COSINE_ALPHA:.0e})")
print(f"  Dropout: {DROPOUT_RATE}")
print(f"  Label Smoothing: {LABEL_SMOOTHING}")


# %%
# =========================
# 4. Compile — CosineDecay
# =========================
steps_per_epoch = math.ceil(len(X_train) / BATCH_SIZE)
decay_steps     = steps_per_epoch * EPOCHS

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=FINETUNE_LR,  # 5e-6
    decay_steps=decay_steps,
    alpha=COSINE_ALPHA                  # final LR = 5e-6 × 0.02 = 1e-7
)
print(f"\n[CosineDecay]")
print(f"  steps_per_epoch: {steps_per_epoch}")
print(f"  decay_steps:     {decay_steps}")
print(f"  initial LR:      {FINETUNE_LR}")
print(f"  final LR:        {FINETUNE_LR * COSINE_ALPHA:.2e}")

optimizer = Adam(learning_rate=lr_schedule)
loss_fn   = CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
print(f"Compiled. CosineDecay {FINETUNE_LR}→{FINETUNE_LR*COSINE_ALPHA:.0e}, LS={LABEL_SMOOTHING}")


# %%
# =========================
# 5. Callback: MF monitoring
# =========================
class EpochMFLogger(tf.keras.callbacks.Callback):
    def __init__(self, mf1_ds, mf3_ds):
        super().__init__()
        self.mf1_ds = mf1_ds; self.mf3_ds = mf3_ds
        self.mf1_acc=[]; self.mf3_acc=[]; self.mf1_loss=[]; self.mf3_loss=[]
        self.epoch_records=[]; self.log_lines=[]
        self._t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mf1_loss, mf1_acc = self.model.evaluate(self.mf1_ds, verbose=0)
        mf3_loss, mf3_acc = self.model.evaluate(self.mf3_ds, verbose=0)
        self.mf1_loss.append(float(mf1_loss)); self.mf1_acc.append(float(mf1_acc))
        self.mf3_loss.append(float(mf3_loss)); self.mf3_acc.append(float(mf3_acc))
        elapsed = time.time() - self._t0 if self._t0 else np.nan
        record = {
            "epoch":          epoch + 1,
            "train_accuracy": float(logs.get("accuracy",     np.nan)),
            "train_loss":     float(logs.get("loss",         np.nan)),
            "val_accuracy":   float(logs.get("val_accuracy", np.nan)),
            "val_loss":       float(logs.get("val_loss",     np.nan)),
            "mf1_accuracy":   float(mf1_acc), "mf1_loss": float(mf1_loss),
            "mf3_accuracy":   float(mf3_acc), "mf3_loss": float(mf3_loss),
            "epoch_seconds":  float(elapsed),
        }
        self.epoch_records.append(record)
        line = (f"Epoch {epoch+1}/{self.params.get('epochs','?')} | "
                f"train_acc={record['train_accuracy']:.4f}, train_loss={record['train_loss']:.4f}, "
                f"val_acc={record['val_accuracy']:.4f}, val_loss={record['val_loss']:.4f}, "
                f"mf1_acc={record['mf1_accuracy']:.4f}, mf3_acc={record['mf3_accuracy']:.4f}, "
                f"time={record['epoch_seconds']:.1f}s")
        self.log_lines.append(line)
        print(f" - MF1_acc: {mf1_acc:.4f} | MF3_acc: {mf3_acc:.4f}")

mf_logger = EpochMFLogger(mf1_ds, mf3_ds)


# %%
# =========================
# 6. Checkpoint
# =========================
best_valloss_path = out_path("best_model_valloss.keras")
best_valacc_path  = out_path("best_model_valacc.keras")
final_model_path  = out_path("final_model.keras")

ckpt_loss = ModelCheckpoint(
    filepath=best_valloss_path,
    monitor="val_loss", save_best_only=True, mode="min", verbose=1
)
ckpt_acc = ModelCheckpoint(
    filepath=best_valacc_path,
    monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
)


# %%
# =========================
# 7. Model training
# =========================
print(f"\n[Training Start] EfficientNet-B3 G6a")
print(f"  IMG=180, CosineDecay {FINETUNE_LR}→{FINETUNE_LR*COSINE_ALPHA:.0e}, unfreeze={UNFREEZE_LAST_N}, epochs={EPOCHS}")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[ckpt_loss, ckpt_acc, mf_logger],
    verbose=1
)


# %%
# =========================
# 8. Save final model
# =========================
model.save(final_model_path)
print(f"Saved final model: {final_model_path}")


# %%
# =========================
# 9. Save training logs
# =========================
history_df  = pd.DataFrame(mf_logger.epoch_records)
history_csv = out_path("training_history.csv")
history_df.to_csv(history_csv, index=False)

best_ep_valloss = int(history_df["val_loss"].idxmin() + 1)
best_ep_valacc  = int(history_df["val_accuracy"].idxmax() + 1)

with open(out_path("training_log.txt"), "w", encoding="utf-8") as f:
    f.write(f"RUN_NAME: {RUN_NAME}\n")
    f.write(f"BACKBONE: EfficientNetB3 (ImageNet pretrained)\n")
    f.write(f"UNFREEZE_LAST_N: {UNFREEZE_LAST_N} (~{UNFREEZE_LAST_N/len(base_model.layers)*100:.1f}%)\n")
    f.write(f"FINETUNE_LR: {FINETUNE_LR}\n")
    f.write(f"DROPOUT: {DROPOUT_RATE}\n")
    f.write(f"LABEL_SMOOTHING: {LABEL_SMOOTHING}\n")
    f.write(f"CLS1_UNDERSAMPLE_COUNT: {CLS1_UNDERSAMPLE_COUNT}\n")
    f.write(f"EPOCHS: {EPOCHS}\n\n")
    f.write("[Epoch Logs]\n")
    for line in mf_logger.log_lines:
        f.write(line + "\n")
    f.write(f"\n[Best val_loss] epoch={best_ep_valloss}, "
            f"val_loss={history_df.loc[best_ep_valloss-1,'val_loss']:.4f}\n")
    f.write(f"[Best val_acc]  epoch={best_ep_valacc}, "
            f"val_acc={history_df.loc[best_ep_valacc-1,'val_accuracy']:.4f}\n")

print(f"Saved training history: {history_csv}")


# %%
# =========================
# 10. Save and display training curves
# =========================
def save_curve(x, curves, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    for y, label in curves:
        plt.plot(x, y, label=label)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.legend(); plt.grid(); plt.tight_layout()
    plt.savefig(save_path, dpi=200); plt.close()

epochs_arr = history_df["epoch"].values

save_curve(epochs_arr,
    [(history_df["train_accuracy"].values, "Train Acc."),
     (history_df["val_accuracy"].values,   "Val Acc.")],
    f"Accuracy Curve ({RUN_NAME})", "Epoch", "Accuracy",
    out_path("accuracy_curve.png"))

save_curve(epochs_arr,
    [(history_df["mf1_accuracy"].values, "MF1 Acc."),
     (history_df["mf3_accuracy"].values, "MF3 Acc.")],
    f"Accuracy Curve MF ({RUN_NAME})", "Epoch", "Accuracy",
    out_path("accuracy_curve_mf.png"))

save_curve(epochs_arr,
    [(history_df["train_accuracy"].values, "Train Acc."),
     (history_df["val_accuracy"].values,   "Val Acc."),
     (history_df["mf1_accuracy"].values,   "MF1 Acc."),
     (history_df["mf3_accuracy"].values,   "MF3 Acc.")],
    f"Accuracy Curve All ({RUN_NAME})", "Epoch", "Accuracy",
    out_path("accuracy_curve_all.png"))

save_curve(epochs_arr,
    [(history_df["train_loss"].values, "Train Loss"),
     (history_df["val_loss"].values,   "Val Loss")],
    f"Loss Curve ({RUN_NAME})", "Epoch", "Loss",
    out_path("loss_curve.png"))

print("Saved curve figures.")

# Display curves when running interactively
plt.figure(figsize=(10, 6))
plt.plot(epochs_arr, history_df["train_accuracy"], label="Train Acc.")
plt.plot(epochs_arr, history_df["val_accuracy"],   label="Val Acc.")
plt.title(f"Accuracy Curve ({RUN_NAME})")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_arr, history_df["train_loss"], label="Train Loss")
plt.plot(epochs_arr, history_df["val_loss"],   label="Val Loss")
plt.title(f"Loss Curve ({RUN_NAME})")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 6))
plt.plot(epochs_arr, history_df["mf1_accuracy"], label="MF1 Acc.")
plt.plot(epochs_arr, history_df["mf3_accuracy"], label="MF3 Acc.")
plt.title(f"MF Accuracy Curve ({RUN_NAME})")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()


# %%
# =========================
# 11. Evaluation utilities
# =========================
def predict_dataset(model, ds):
    probs = model.predict(ds, verbose=0)
    return np.argmax(probs, axis=1), np.max(probs, axis=1), probs

def save_cm_figure(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names))); ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(save_path, dpi=200); plt.close(fig)

def evaluate_subset(model, model_name, subset_name, ds, y_true, paths, class_names):
    y_pred, confs, probs = predict_dataset(model, ds)
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv = out_path(f"{model_name}_{subset_name}_confusion_matrix.csv")
    cm_png = out_path(f"{model_name}_{subset_name}_confusion_matrix.png")
    cm_df.to_csv(cm_csv)
    save_cm_figure(cm, class_names,
        f"{model_name.upper()} - {subset_name.upper()} Confusion Matrix", cm_png)

    report_df = pd.DataFrame(
        classification_report(y_true, y_pred, target_names=class_names,
                              output_dict=True, zero_division=0)
    ).transpose()
    report_df.to_csv(out_path(f"{model_name}_{subset_name}_classification_report.csv"))

    pd.DataFrame({
        "class_id":   np.arange(NUM_CLASSES),
        "class_name": class_names,
        "true_count": [(y_true == i).sum() for i in range(NUM_CLASSES)],
        "pred_count": [(y_pred == i).sum() for i in range(NUM_CLASSES)],
    }).to_csv(out_path(f"{model_name}_{subset_name}_prediction_distribution.csv"), index=False)

    mis_mask = y_true != y_pred
    pd.DataFrame({
        "filepath":        paths[mis_mask],
        "true_label":      y_true[mis_mask],
        "pred_label":      y_pred[mis_mask],
        "true_name":       [class_names[i] for i in y_true[mis_mask]],
        "pred_name":       [class_names[i] for i in y_pred[mis_mask]],
        "pred_confidence": confs[mis_mask],
    }).sort_values("pred_confidence", ascending=False).to_csv(
        out_path(f"{model_name}_{subset_name}_misclassified.csv"), index=False)

    off_diag = cm.copy(); np.fill_diagonal(off_diag, 0)
    confusion_pairs = [
        {"true_class": i, "pred_class": j,
         "true_name": class_names[i], "pred_name": class_names[j],
         "count": int(off_diag[i, j])}
        for i in range(NUM_CLASSES) for j in range(NUM_CLASSES)
        if i != j and off_diag[i, j] > 0
    ]
    pd.DataFrame(
        confusion_pairs if confusion_pairs else
        [{"true_class":0,"pred_class":0,"true_name":"","pred_name":"","count":0}]
    ).sort_values("count", ascending=False).to_csv(
        out_path(f"{model_name}_{subset_name}_top_confusions.csv"), index=False)

    return {
        "model_name":           model_name,
        "subset":               subset_name,
        "n_samples":            int(len(y_true)),
        "accuracy":             float(acc),
        "macro_precision":      float(macro_p),
        "macro_recall":         float(macro_r),
        "macro_f1":             float(macro_f1),
        "weighted_precision":   float(weighted_p),
        "weighted_recall":      float(weighted_r),
        "weighted_f1":          float(weighted_f1),
        "confusion_matrix_csv": cm_csv,
        "confusion_matrix_png": cm_png,
    }


# %%
# =========================
# 12. Evaluate best and final models
# =========================
best_loss_model = tf.keras.models.load_model(best_valloss_path)
best_acc_model  = tf.keras.models.load_model(best_valacc_path)
final_model_obj = tf.keras.models.load_model(final_model_path)
print("Loaded best_valloss, best_valacc, final models.")

evaluation_rows = []
for m, mname in [(best_loss_model, "best_valloss"),
                 (best_acc_model,  "best_valacc"),
                 (final_model_obj, "final_model")]:
    evaluation_rows.append(evaluate_subset(
        m, mname, "mf1", mf1_ds, y_test_mf1, X_test_mf1, CLASS_NAMES))
    evaluation_rows.append(evaluate_subset(
        m, mname, "mf3", mf3_ds, y_test_mf3, X_test_mf3, CLASS_NAMES))

eval_df = pd.DataFrame(evaluation_rows)
eval_df.to_csv(out_path("evaluation_summary.csv"), index=False)
print("Saved evaluation summary.")


# %%
# =========================
# 13. Diagnostic summary
# =========================
def safe_float(x):
    return float(x) if x is not None and not pd.isna(x) else np.nan

final_train = safe_float(history_df["train_accuracy"].iloc[-1])
final_val   = safe_float(history_df["val_accuracy"].iloc[-1])
final_vl    = safe_float(history_df["val_loss"].iloc[-1])
best_val    = safe_float(history_df["val_accuracy"].max())
best_vl     = safe_float(history_df["val_loss"].min())

best_mf1 = eval_df[(eval_df["model_name"]=="best_valacc") &
                   (eval_df["subset"]=="mf1")]["accuracy"].iloc[0]
best_mf3 = eval_df[(eval_df["model_name"]=="best_valacc") &
                   (eval_df["subset"]=="mf3")]["accuracy"].iloc[0]

diag_lines = [
    f"RUN_NAME: {RUN_NAME}",
    f"BACKBONE: EfficientNetB3 (ImageNet pretrained)",
    f"UNFREEZE_LAST_N: {UNFREEZE_LAST_N} (~{UNFREEZE_LAST_N/len(base_model.layers)*100:.1f}%)",
    f"FINETUNE_LR: {FINETUNE_LR}",
    f"SCHEDULER: CosineDecay (alpha={COSINE_ALPHA}, final={FINETUNE_LR*COSINE_ALPHA:.2e})",
    f"DECAY_STEPS: {decay_steps} (steps_per_epoch={steps_per_epoch} × epochs={EPOCHS})",
    f"IMG_SIZE: {IMG_SIZE} (180px)",
    f"DROPOUT: {DROPOUT_RATE}",
    f"LABEL_SMOOTHING: {LABEL_SMOOTHING}",
    f"CLS1_UNDERSAMPLE_COUNT: {CLS1_UNDERSAMPLE_COUNT}",
    f"EPOCHS: {EPOCHS}",
    "",
    "[Training Summary]",
    f"Final train_acc: {final_train:.4f}",
    f"Final val_acc:   {final_val:.4f}",
    f"Final val_loss:  {final_vl:.4f}",
    f"Best val_acc:    {best_val:.4f} (epoch {best_ep_valacc})",
    f"Best val_loss:   {best_vl:.4f} (epoch {best_ep_valloss})",
    "",
    "[Best Model (best_valacc) Test Summary]",
    f"MF1 accuracy: {best_mf1:.4f}",
    f"MF3 accuracy: {best_mf3:.4f}",
    "",
    "[Learning Diagnosis]",
    f"Train-Val gap (final): {final_train - final_val:.4f}",
    "",
    "[Fine-tuning Config]",
    f"  Backbone: EfficientNetB3, top-{UNFREEZE_LAST_N} layers unfrozen",
    f"  BN freeze: full model scope",
    f"  Head trainability: default trainable setting retained",
    f"  IMG_SIZE: {IMG_SIZE} (G5b 180px)",
    f"  LR: {FINETUNE_LR}",
    f"  Scheduler: CosineDecay (initial={FINETUNE_LR}, final={FINETUNE_LR*COSINE_ALPHA:.2e})",
    f"  Dropout: {DROPOUT_RATE}",
    f"  Label Smoothing: {LABEL_SMOOTHING}",
    f"  Head: Dense(512) → Dropout({DROPOUT_RATE}) → Dense({NUM_CLASSES})",
    "",
    "[Recommended Paper Figures]",
    f"1. {os.path.basename(out_path('accuracy_curve.png'))}",
    f"2. {os.path.basename(out_path('accuracy_curve_mf.png'))}",
    f"3. {os.path.basename(out_path('accuracy_curve_all.png'))}",
    f"4. {os.path.basename(out_path('loss_curve.png'))}",
    f"5. {os.path.basename(out_path('best_valacc_mf1_confusion_matrix.png'))}",
    f"6. {os.path.basename(out_path('best_valacc_mf3_confusion_matrix.png'))}",
]

with open(out_path("diagnostic_summary.txt"), "w", encoding="utf-8") as f:
    for line in diag_lines:
        f.write(line + "\n")
print("Saved diagnostic summary.")


# %%
print("\nEfficientNet-B3 G6a training complete.")
print(f"  IMG=180, CosineDecay {FINETUNE_LR}→{FINETUNE_LR*COSINE_ALPHA:.0e}, unfreeze={UNFREEZE_LAST_N}, epochs={EPOCHS}")
print(f"Best (val_loss): {best_valloss_path}")
print(f"Best (val_acc):  {best_valacc_path}")
print(f"Final model:     {final_model_path}")
