"""
Generate spatial reconstruction figures for MF1 and MF3 test images.

This script overlays ground-truth labels, model predictions, and classification
errors onto the original thin-section images using tile coordinates stored in
.dat files.
"""


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Patch

Image.MAX_IMAGE_PIXELS = None

# ==================================================
# Paths
# ==================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(REPO_DIR, "data")

LV1_DIR = os.path.join(DATA_DIR, "DAT", "LV1")
LV2_DIR = os.path.join(DATA_DIR, "DAT", "LV2_cal")

RESULT_DIR = os.path.join(REPO_DIR, "results")

MF1_CSV = os.path.join(
    RESULT_DIR,
    "efficientnetb3_finetune_v6a_onestop_best_valacc_mf1_all_predictions.csv"
)

MF3_CSV = os.path.join(
    RESULT_DIR,
    "efficientnetb3_finetune_v6a_onestop_best_valacc_mf3_all_predictions.csv"
)

OUT_DIR = os.path.join(
    RESULT_DIR,
    "Figures5_6_overlay_2x2"
)

os.makedirs(OUT_DIR, exist_ok=True)

# ==================================================
# Settings
# ==================================================

TILE_SIZE = 104
GRID_CELL_SIZE = 120

ALPHA_LABEL = 0.45
ALPHA_ERROR = 0.30

CLASS_NAMES = {
    0: "Intraclast",
    1: "Girvanella crust",
    2: "Thromboid",
    3: "Microbial peloid",
    4: "Peloid",
    5: "Coated grain",
    6: "Bioclast",
    7: "Quartz",
    8: "Micrite",
    9: "Cement",
}

CLASS_COLORS = (
    plt.cm.tab10(np.arange(10))[:, :3] * 255
).astype(np.uint8)

ERROR_COLOR = np.array([200, 80, 80], dtype=np.uint8)

# ==================================================
# Load predictions
# ==================================================

mf1_pred_df = pd.read_csv(MF1_CSV)
mf3_pred_df = pd.read_csv(MF3_CSV)

print("\nDetected MF1 samples:")
for s in sorted(mf1_pred_df["sample"].unique()):
    print(" -", s)

print("\nDetected MF3 samples:")
for s in sorted(mf3_pred_df["sample"].unique()):
    print(" -", s)

# ==================================================
# Figure configuration
# ==================================================
# Figure 5: MF1 result reconstructed by combining two MF1 annotation sets.
# Figure 6: MF3 result reconstructed from one MF3 test sample.

FIGURE_CONFIGS = [
    {
        "figure": "Figure 5",
        "subset": "MF1",
        "title": "MF1 test image",
        "samples": [
            "(cali.) 20150212 HXND 2-3 (1)",
            "(cali.) 20150212 HXND 2-3 (2)",
        ],
        "pred_df": mf1_pred_df,
        "base_sample": "(cali.) 20150212 HXND 2-3 (1)",
        "out_name": "Figure05_MF1_overlay_2x2",
        "figsize": (9.5, 11),
        "wspace": -0.10,
        "hspace": 0.18,
        "left": 0.08,
        "right": 0.92,
        "top": 0.88,
        "bottom": 0.15,
    },
    {
        "figure": "Figure 6",
        "subset": "MF3",
        "title": "MF3 test image",
        "samples": [
            "(edit) Unit 1 1-5(1)(3,1)",
        ],
        "pred_df": mf3_pred_df,
        "base_sample": "(edit) Unit 1 1-5(1)(3,1)",
        "out_name": "Figure06_MF3_overlay_2x2",
        "figsize": (11.5, 11),
        "wspace": 0.00,
        "hspace": 0.18,
        "left": 0.04,
        "right": 0.96,
        "top": 0.88,
        "bottom": 0.15,
    },
]

# ==================================================
# Helper functions
# ==================================================

def find_dat(sample):
    path = os.path.join(
        LV2_DIR,
        f"{sample}_grid&rock.png.dat"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return path


def find_rock(sample):
    folder = os.path.join(
        LV1_DIR,
        sample
    )

    if not os.path.exists(folder):
        raise FileNotFoundError(folder)

    candidates = []

    for f in os.listdir(folder):
        lower = f.lower()

        if "rock" in lower and lower.endswith(".png"):
            candidates.append(os.path.join(folder, f))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No rock image found in {folder}"
        )

    candidates = sorted(candidates)
    return candidates[0]


def load_dat(dat_path):
    df = pd.read_csv(
        dat_path,
        sep=r"\s+",
        header=None,
        names=[
            "tile_index",
            "y0",
            "x0",
            "green_sum"
        ]
    )

    df["tile_index"] = df["tile_index"].astype(int)
    df["y0"] = df["y0"].astype(int)
    df["x0"] = df["x0"].astype(int)

    return df


def read_rgb_image(path):
    return np.array(
        Image.open(path).convert("RGB")
    )


def blend_patch(roi, color, alpha):
    patch = np.zeros_like(roi, dtype=np.uint8)
    patch[:] = color

    blended = (
        roi.astype(np.float32) * (1 - alpha)
        + patch.astype(np.float32) * alpha
    )

    return np.clip(blended, 0, 255).astype(np.uint8)


def build_overlay(img, merged, label_column):
    overlay = img.copy()

    h, w = img.shape[:2]

    offset_y = (GRID_CELL_SIZE - TILE_SIZE) // 2
    offset_x = (GRID_CELL_SIZE - TILE_SIZE) // 2

    skipped = 0

    for _, row in merged.iterrows():
        y = int(row["y0"]) + offset_y
        x = int(row["x0"]) + offset_x

        if y < 0 or x < 0:
            skipped += 1
            continue

        if y + TILE_SIZE > h or x + TILE_SIZE > w:
            skipped += 1
            continue

        cls = int(row[label_column])
        color = CLASS_COLORS[cls]

        roi = overlay[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ]

        overlay[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ] = blend_patch(
            roi=roi,
            color=color,
            alpha=ALPHA_LABEL
        )

    return overlay, skipped


def build_error_overlay(img, merged):
    overlay = img.copy()

    h, w = img.shape[:2]

    offset_y = (GRID_CELL_SIZE - TILE_SIZE) // 2
    offset_x = (GRID_CELL_SIZE - TILE_SIZE) // 2

    skipped = 0
    n_errors = 0

    for _, row in merged.iterrows():
        if int(row["true_label"]) == int(row["pred_label"]):
            continue

        y = int(row["y0"]) + offset_y
        x = int(row["x0"]) + offset_x

        if y < 0 or x < 0:
            skipped += 1
            continue

        if y + TILE_SIZE > h or x + TILE_SIZE > w:
            skipped += 1
            continue

        roi = overlay[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ]

        overlay[
            y:y + TILE_SIZE,
            x:x + TILE_SIZE
        ] = blend_patch(
            roi=roi,
            color=ERROR_COLOR,
            alpha=ALPHA_ERROR
        )

        n_errors += 1

    return overlay, skipped, n_errors


def make_class_legend_handles():
    handles = []

    for i in range(10):
        color = CLASS_COLORS[i] / 255.0
        handles.append(
            Patch(
                facecolor=color,
                edgecolor="none",
                label=f"{i}: {CLASS_NAMES[i]}"
            )
        )

    handles.append(
        Patch(
            facecolor=(1, 0, 0, ALPHA_ERROR),
            edgecolor="none",
            label="Error: incorrect prediction"
        )
    )

    return handles


def merge_samples_for_figure(config):
    merged_list = []

    for sample in config["samples"]:
        print(f"  Merging sample: {sample}")

        sample_pred = config["pred_df"][
            config["pred_df"]["sample"] == sample
        ].copy()

        if len(sample_pred) == 0:
            raise ValueError(f"No predictions found for sample: {sample}")

        dat_path = find_dat(sample)
        coord_df = load_dat(dat_path)

        merged = pd.merge(
            coord_df,
            sample_pred,
            on="tile_index",
            how="inner"
        )

        if len(merged) == 0:
            raise ValueError(f"No matched tiles for sample: {sample}")

        merged["source_sample"] = sample
        merged["source_dat_path"] = dat_path

        merged_list.append(merged)

        print(f"    matched tiles: {len(merged)}")

    combined = pd.concat(
        merged_list,
        axis=0,
        ignore_index=True
    )

    duplicated = combined.duplicated(
        subset=["source_sample", "tile_index"]
    ).sum()

    if duplicated > 0:
        print(f"[Warning] duplicated source_sample + tile_index rows: {duplicated}")

    return combined


def save_overlay_figure(config):
    print(f"\nProcessing {config['figure']} ({config['subset']})")

    merged = merge_samples_for_figure(config)

    rock_path = find_rock(config["base_sample"])
    img = read_rgb_image(rock_path)

    true_overlay, skipped_true = build_overlay(
        img=img,
        merged=merged,
        label_column="true_label"
    )

    pred_overlay, skipped_pred = build_overlay(
        img=img,
        merged=merged,
        label_column="pred_label"
    )

    error_overlay, skipped_error, n_errors = build_error_overlay(
        img=img,
        merged=merged
    )

    accuracy = (
        merged["true_label"].astype(int)
        == merged["pred_label"].astype(int)
    ).mean()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=config["figsize"]
    )

    axes = axes.ravel()

    axes[0].imshow(img)
    axes[0].set_title("(a) Original", fontsize=11, fontweight="normal", pad=12)

    axes[1].imshow(true_overlay)
    axes[1].set_title("(b) Ground truth", fontsize=11, fontweight="normal", pad=12)

    axes[2].imshow(pred_overlay)
    axes[2].set_title("(c) Prediction", fontsize=11, fontweight="normal", pad=12)

    axes[3].imshow(error_overlay)
    axes[3].set_title("(d) Errors", fontsize=11, fontweight="normal", pad=12)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        config["title"],
        fontsize=13,
        fontweight="normal",
        y=0.985
    )

    legend_handles = make_class_legend_handles()

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015)
    )

    plt.subplots_adjust(
        left=config["left"],
        right=config["right"],
        top=config["top"],
        bottom=config["bottom"],
        wspace=config["wspace"],
        hspace=config["hspace"]
    )

    out_base = os.path.join(
        OUT_DIR,
        config["out_name"]
    )

    out_png = out_base + ".png"
    out_tif = out_base + ".tif"
    out_pdf = out_base + ".pdf"

    plt.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        out_tif,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        out_pdf,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Base rock image:", rock_path)
    print("Matched tiles:", len(merged))
    print("Errors:", n_errors)
    print("Accuracy:", f"{accuracy * 100:.2f}%")
    print("Saved:", out_png)

    return {
        "figure": config["figure"],
        "subset": config["subset"],
        "title": config["title"],
        "samples": "; ".join(config["samples"]),
        "base_sample": config["base_sample"],
        "rock_path": rock_path,
        "matched_tiles": len(merged),
        "errors": n_errors,
        "accuracy": accuracy,
        "skipped_true": skipped_true,
        "skipped_pred": skipped_pred,
        "skipped_error": skipped_error,
        "output_png": out_png,
        "output_tif": out_tif,
        "output_pdf": out_pdf
    }

# ==================================================
# Main execution
# ==================================================

summary_rows = []

for config in FIGURE_CONFIGS:
    summary_rows.append(
        save_overlay_figure(config)
    )

summary_df = pd.DataFrame(summary_rows)

summary_path = os.path.join(
    OUT_DIR,
    "Figures5_6_overlay_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nDone.")
print("Summary saved:", summary_path)
display(summary_df)
