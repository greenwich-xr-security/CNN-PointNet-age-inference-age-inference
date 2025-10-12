import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None
import pandas as pd
import cv2

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    tk = None
    ttk = None

# === CONFIG ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "resnet18_hands_multilabel.pth"

ROOT_FOLDER = Path(r"C:\Users\Staff\Documents\HandsDatasets\archive\Photos")
CSV_FILE = Path(r"C:\Users\Staff\Documents\HandsDatasets\archive\Dataset Details.csv")
OUTPUT_CSV = Path(r"C:\Users\Staff\Documents\HandsDatasets\archive\annotated_dataset_details.csv")

SHOW_INTERACTIVE = True
AUTO_CONFIDENCE_THRESHOLD = 0.95

# === LABELS ===
CLASS_LABELS = [
    "dorsal left",
    "dorsal right",
    "palmar left",
    "palmar right",
]
dorsal_palmar_map = {
    0: "Dorsal",
    1: "Palmar",
}
left_right_map = {
    0: "Left",
    1: "Right",
}

# === TRANSFORMS (same as during training) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def build_image_key(person_no, age, gender, photo_no):
    """Create a stable key for a metadata row so we can skip repeats."""
    parts = [int(person_no), int(age)]
    if gender is not None and gender != "" and not pd.isna(gender):
        parts.append(int(gender))
    parts.append(int(photo_no))
    return "_".join(str(p) for p in parts)


# === MODEL ===
class ResNetMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dorsal_palmar_head = nn.Linear(num_features, 2)
        self.left_right_head = nn.Linear(num_features, 2)

    def forward(self, x):
        features = self.backbone(x)
        dorsal_palmar_logits = self.dorsal_palmar_head(features)
        left_right_logits = self.left_right_head(features)
        return dorsal_palmar_logits, left_right_logits


class AnnotationUI:
    def __init__(self, class_labels):
        if tk is None or ttk is None or ImageTk is None:
            raise RuntimeError("Tkinter/ImageTk is not available on this system")
        self.class_labels = class_labels
        self.root = tk.Tk()
        self.root.title("Hand Annotation")
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        self.image_label = ttk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)

        self.info_var = tk.StringVar()
        self.pred_var = tk.StringVar()
        self.progress_var = tk.StringVar()

        info_frame = ttk.Frame(self.root)
        info_frame.pack(padx=10, pady=(0, 10))
        tk.Label(info_frame, textvariable=self.info_var, justify="center").pack()
        tk.Label(info_frame, textvariable=self.pred_var, justify="center", fg="blue").pack()
        tk.Label(info_frame, textvariable=self.progress_var, justify="center", fg="green").pack()

        self.button_frame = ttk.Frame(self.root)
        self.button_frame.pack(pady=5)

        self.correct_btn = ttk.Button(self.button_frame, text="Correct", command=self._on_correct)
        self.correct_btn.grid(row=0, column=0, padx=5)
        self.incorrect_btn = ttk.Button(self.button_frame, text="Incorrect", command=self._on_incorrect)
        self.incorrect_btn.grid(row=0, column=1, padx=5)
        self.skip_btn = ttk.Button(self.button_frame, text="Skip", command=self._on_skip)
        self.skip_btn.grid(row=0, column=2, padx=5)
        self.quit_btn = ttk.Button(self.button_frame, text="Quit", command=self._on_quit)
        self.quit_btn.grid(row=0, column=3, padx=5)

        self.label_frame = ttk.Frame(self.root)
        self.label_buttons = []
        for idx, label in enumerate(self.class_labels):
            btn = ttk.Button(self.label_frame, text=label.title(), command=lambda l=label: self._on_label_selected(l))
            btn.grid(row=idx // 2, column=idx % 2, padx=5, pady=5, sticky="ew")
            self.label_buttons.append(btn)

        for col in range(2):
            self.label_frame.columnconfigure(col, weight=1)

        self.choice = None
        self.selected_label = None
        self.predicted_label = None

    def annotate(self, img_bgr, info_text, prediction_label, progress_text):
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image  # keep reference

        self.info_var.set(info_text)
        self.pred_var.set(
            f"Prediction: {prediction_label}" if prediction_label != "Unknown" else "Prediction unavailable"
        )
        self.progress_var.set(progress_text)

        self.choice = None
        self.selected_label = None
        self.predicted_label = prediction_label.lower() if prediction_label != "Unknown" else None

        self._set_label_buttons_visible(False)
        self.correct_btn.configure(state=tk.NORMAL if self.predicted_label else tk.DISABLED)

        while self.choice is None:
            self.root.update_idletasks()
            self.root.update()

        if self.choice == "quit":
            return "quit", None
        if self.choice == "skip":
            return "skip", None
        if self.choice == "correct":
            return "correct", self.predicted_label
        if self.choice == "incorrect":
            return "incorrect", self.selected_label
        return "skip", None

    def destroy(self):
        self.root.destroy()

    def _on_correct(self):
        self.choice = "correct"

    def _on_incorrect(self):
        self._set_label_buttons_visible(True)

    def _on_skip(self):
        self.choice = "skip"

    def _on_quit(self):
        self.choice = "quit"

    def _on_label_selected(self, label):
        self.selected_label = label
        self.choice = "incorrect"

    def _set_label_buttons_visible(self, visible):
        if visible:
            self.label_frame.pack(pady=(0, 10))
        else:
            self.label_frame.pack_forget()
            self.selected_label = None


model = ResNetMultiHead()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=True)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully on", DEVICE)

annotation_ui = None
if SHOW_INTERACTIVE and tk is not None and ttk is not None:
    try:
        annotation_ui = AnnotationUI(CLASS_LABELS)
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to initialise GUI annotation (falling back to console): {exc}")
        annotation_ui = None
elif SHOW_INTERACTIVE:
    print("Tkinter not available; falling back to console prompts.")

# === LOAD CSV METADATA ===
raw_df = pd.read_csv(CSV_FILE, sep=';')
processed_records = []
for _, row in raw_df.iterrows():
    try:
        person_no = int(row['Person No'])
        age = int(row['Age']) if pd.notna(row['Age']) else -1
        gender_raw = row.get('Gender', None)
        gender = int(gender_raw) if pd.notna(gender_raw) else None
        photo_no = int(row['Photo No'])
    except (ValueError, TypeError, KeyError):
        continue

    image_key = build_image_key(person_no, age, gender, photo_no)
    record = row.to_dict()
    record['Person No'] = person_no
    record['Age'] = age
    record['Gender'] = gender if gender is not None else pd.NA
    record['Photo No'] = photo_no
    record['imageKey'] = image_key
    processed_records.append(record)

if not processed_records:
    print("No valid entries found in metadata; nothing to annotate.")
    if annotation_ui:
        annotation_ui.destroy()
    raise SystemExit(0)

meta_df = pd.DataFrame(processed_records)
meta_df = meta_df.drop_duplicates(subset='imageKey').sample(frac=1, random_state=None).reset_index(drop=True)
unique_keys = meta_df['imageKey'].tolist()
unique_key_set = set(unique_keys)
total_entries = len(unique_keys)

existing_annotations = set()
csv_already_initialised = False
if OUTPUT_CSV.exists():
    try:
        existing_df = pd.read_csv(OUTPUT_CSV)
        if not existing_df.empty:
            csv_already_initialised = True
            if 'imageKey' in existing_df.columns:
                existing_annotations.update(existing_df['imageKey'].dropna().astype(str))
            else:
                for _, past_row in existing_df.iterrows():
                    try:
                        key = build_image_key(
                            int(past_row.get('Person No')),
                            int(past_row.get('Age', -1)),
                            past_row.get('Gender', None),
                            int(past_row.get('Photo No')),
                        )
                    except Exception:  # noqa: BLE001
                        continue
                    existing_annotations.add(key)
            existing_annotations &= unique_key_set
            print(f"Loaded {len(existing_annotations)} existing annotations from {OUTPUT_CSV}.")
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read existing annotations: {exc}. Starting fresh.")
        csv_already_initialised = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 0
else:
    print("No existing annotation file found; a new one will be created.")

if total_entries == 0:
    print("No entries available for annotation after processing; nothing to do.")
    if annotation_ui:
        annotation_ui.destroy()
    raise SystemExit(0)

initial_progress = (len(existing_annotations) / total_entries) * 100 if total_entries else 0
print(
    f"Starting coverage: {initial_progress:.1f}% "
    f"({len(existing_annotations)} / {total_entries})."
)

new_annotations = 0
auto_annotations = 0
if annotation_ui:
    print("Interactive GUI mode enabled. Use window buttons to annotate.")
else:
    print("Console annotation mode. Respond to prompts in terminal.")
    print("Options: [c]orrect, [i]ncorrect, [s]kip, [q]uit.")

user_exit = False
for idx, (_, row) in enumerate(meta_df.iterrows(), start=1):
    person_no = int(row['Person No'])
    age = int(row['Age'])
    gender_val = row['Gender']
    gender = None if pd.isna(gender_val) else int(gender_val)
    photo_no = int(row['Photo No'])
    image_key = row['imageKey']

    if image_key in existing_annotations:
        progress_pct = (len(existing_annotations) / total_entries) * 100 if total_entries else 0
        print(
            f"[{idx}/{len(meta_df)}] entry already annotated; skipping. "
            f"Progress: {progress_pct:.1f}%"
        )
        continue

    parts = [person_no, age]
    if gender is not None:
        parts.append(gender)
    parts.append(photo_no)
    base_name = "_".join(str(p) for p in parts)

    found = False
    for ext in (".jpg", ".png", ".jpeg"):
        img_path = ROOT_FOLDER / f"{base_name}{ext}"
        if img_path.is_file():
            found = True
            break
    if not found:
        continue

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:  # noqa: BLE001
        print(f"Error loading {img_path}: {e}")
        continue

    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dp_logits, lr_logits = model(img_t)
        dp_probs = F.softmax(dp_logits, dim=1)
        lr_probs = F.softmax(lr_logits, dim=1)
        dp_pred = torch.argmax(dp_probs, dim=1).item()
        lr_pred = torch.argmax(lr_probs, dim=1).item()
        dp_conf = dp_probs[0, dp_pred].item()
        lr_conf = lr_probs[0, lr_pred].item()

    orientation = dorsal_palmar_map.get(dp_pred, "Unknown")
    side = left_right_map.get(lr_pred, "Unknown")
    label = f"{orientation} {side}" if "Unknown" not in (orientation, side) else "Unknown"

    current_progress = (len(existing_annotations) / total_entries) * 100 if total_entries else 0
    auto_label = (
        label != "Unknown"
        and dp_conf >= AUTO_CONFIDENCE_THRESHOLD
        and lr_conf >= AUTO_CONFIDENCE_THRESHOLD
    )

    if auto_label:
        future_progress = ((len(existing_annotations) + 1) / total_entries) * 100 if total_entries else 0
        print(
            f"[{idx}/{len(meta_df)}] {img_path.name} -> Orientation: {orientation} ({dp_conf:.1%}), ",
            f"Side: {side} ({lr_conf:.1%}) | Progress: {future_progress:.1f}% | Auto-annotated"
        )
        final_label = label.lower()
        row_data = row.to_dict()
        row_data['Person No'] = person_no
        row_data['Age'] = age
        row_data['Gender'] = gender if gender is not None else pd.NA
        row_data['Photo No'] = photo_no
        row_data['aspectOfHand'] = final_label
        row_data['imageKey'] = image_key

        annotation_df = pd.DataFrame([row_data])
        annotation_df.to_csv(
            OUTPUT_CSV,
            mode='a',
            index=False,
            header=not csv_already_initialised,
        )
        csv_already_initialised = True
        existing_annotations.add(image_key)
        new_annotations += 1
        auto_annotations += 1
        continue

    print(
        f"[{idx}/{len(meta_df)}] {img_path.name} -> Orientation: {orientation} ({dp_conf:.1%}), ",
        f"Side: {side} ({lr_conf:.1%}) | Progress: {current_progress:.1f}%"
    )

    final_label = None

    if annotation_ui:
        img_cv = cv2.imread(str(img_path))
        if img_cv is None:
            print(f"  Unable to read image via OpenCV: {img_path}")
            continue
        info_text = (
            f"{img_path.name}\n"
            f"Orientation: {orientation} ({dp_conf:.1%})\n"
            f"Side: {side} ({lr_conf:.1%})"
        )
        progress_text = f"Progress: {current_progress:.1f}% ({len(existing_annotations)} / {total_entries})"
        status, selection = annotation_ui.annotate(
            img_cv,
            info_text,
            label,
            progress_text,
        )
        if status == "quit":
            user_exit = True
        elif status in {"correct", "incorrect"}:
            final_label = selection
    else:
        if SHOW_INTERACTIVE:
            print("GUI unavailable; continuing with console prompts.")
        print(f"  Progress: {current_progress:.1f}% ({len(existing_annotations)} / {total_entries})")
        while True:
            user_choice = input("  Mark as [c]orrect, [i]ncorrect, [s]kip, [q]uit: ").strip().lower()
            if user_choice in {"c", "correct"}:
                if label == "Unknown":
                    print("  Prediction unknown; please choose the correct label instead.")
                    continue
                final_label = label.lower()
                break
            if user_choice in {"i", "incorrect"}:
                print("  Select correct label:")
                for option_idx, option_label in enumerate(CLASS_LABELS, start=1):
                    print(f"    {option_idx}. {option_label}")
                while True:
                    selection = input("    Enter number or label: ").strip().lower()
                    if selection.isdigit():
                        selection_idx = int(selection) - 1
                        if 0 <= selection_idx < len(CLASS_LABELS):
                            final_label = CLASS_LABELS[selection_idx]
                            break
                    else:
                        matched = [lbl for lbl in CLASS_LABELS if lbl == selection]
                        if matched:
                            final_label = matched[0]
                            break
                    print("    Invalid selection. Try again.")
                break
            if user_choice in {"s", "skip"}:
                print("  Skipping entry.")
                break
            if user_choice in {"q", "quit"}:
                user_exit = True
                break
            print("  Invalid option. Please respond with c/i/s/q.")

    if user_exit:
        print("Exiting annotation early.")
        break

    if final_label is None:
        continue

    row_data = row.to_dict()
    row_data['Person No'] = person_no
    row_data['Age'] = age
    row_data['Gender'] = gender if gender is not None else pd.NA
    row_data['Photo No'] = photo_no
    row_data['aspectOfHand'] = final_label
    row_data['imageKey'] = image_key

    annotation_df = pd.DataFrame([row_data])
    annotation_df.to_csv(
        OUTPUT_CSV,
        mode='a',
        index=False,
        header=not csv_already_initialised,
    )
    csv_already_initialised = True
    existing_annotations.add(image_key)
    new_annotations += 1

if annotation_ui:
    annotation_ui.destroy()

completion_pct = (len(existing_annotations) / total_entries) * 100 if total_entries else 0
if new_annotations:
    print(
        f"Saved {new_annotations} new annotations to {OUTPUT_CSV}. "
        f"Coverage now {completion_pct:.1f}% ({len(existing_annotations)} / {total_entries})."
    )
else:
    if existing_annotations:
        print(
            "No new annotations were recorded; annotation file already up to date. "
            f"Coverage remains {completion_pct:.1f}% ({len(existing_annotations)} / {total_entries})."
        )
    else:
        print("No annotations were recorded; nothing saved.")
if auto_annotations:
    print(
        f"Auto-annotated {auto_annotations} entries at >= {AUTO_CONFIDENCE_THRESHOLD:.0%} confidence."
    )
