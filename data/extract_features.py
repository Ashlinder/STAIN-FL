import os
import cv2
import numpy as np
import pandas as pd
import torch

from pytorch_i3d import InceptionI3d


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def preprocess_frame(frame):
    # BGR → RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize so shorter edge = 256
    h, w, _ = frame.shape
    scale = 256 / min(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Center crop to 224×224
    h, w, _ = frame.shape
    y = (h - 224) // 2
    x = (w - 224) // 2
    frame = frame[y:y + 224, x:x + 224]

    # Convert to tensor (C,H,W)
    tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor



# Load I3D model
def load_i3d_model(weights_path="models/rgb_imagenet.pt"):
    print("Loading I3D model...")
    model = InceptionI3d(num_classes=400, in_channels=3)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.half()  # FP16 for speed
    model.eval()

    print("I3D model loaded.")
    return model


# Extract 1024D features from a single video
def extract_video_features(video_path, model, clip_len=16):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"WARNING: Could not open {video_path}")
        return None

    frames = []
    clip_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tensor = preprocess_frame(frame)
        frames.append(tensor)

        if len(frames) == clip_len:
            clip = torch.stack(frames, dim=1)  # (3,16,H,W)
            frames = []

            clip = clip.unsqueeze(0)  # (1,3,16,224,224)
            clip = clip.to(device, non_blocking=True).half()

            with torch.no_grad():
                feat = model.extract_features(clip)
                feat = feat.float().cpu().numpy().squeeze()
                clip_features.append(feat)

    cap.release()

    if not clip_features:
        print(f"WARNING: No full 16-frame clips in {video_path}")
        return None

    return np.mean(clip_features, axis=0)  # 1024 features

def extract_all_videos(base_dir=".",
                       output_csv="all_features.csv",
                       output_dir="npy_features"):
    os.makedirs(output_dir, exist_ok=True)

    model = load_i3d_model()

    video_rows = []

    root_paths = {
        "Anomaly": os.path.join(base_dir, "anomaly"),
        "Normal": os.path.join(base_dir, "normal"),
    }

    for label_str, root_folder in root_paths.items():
        if not os.path.exists(root_folder):
            print(f"Skipping missing folder: {root_folder}")
            continue

        print(f"\nProcessing folder: {root_folder}")

        for subdir, _, files in os.walk(root_folder):
            category = os.path.basename(subdir) if label_str == "Anomaly" else "Normal"

            for filename in files:
                if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    continue

                full_path = os.path.join(subdir, filename)
                print(f"  Extracting: {full_path}")

                features = extract_video_features(full_path, model)
                if features is None:
                    continue

                # Save .npy
                base = os.path.splitext(filename)[0]     # removes .mp4 / .avi / .mov / etc.
                npy_name = f"{label_str}_{category}_{base}.npy".replace(" ", "_")
                np.save(os.path.join(output_dir, npy_name), features)

                # Label mapping
                numeric_label = 1 if label_str == "Anomaly" else 0

                # CSV row
                row = {
                    "video_name": filename,
                    "label": numeric_label,
                }
                for i in range(1024):
                    row[f"f{i}"] = features[i]

                video_rows.append(row)

    # Save CSV
    df = pd.DataFrame(video_rows)
    df.to_csv(output_csv, index=False)

    print("\nFeature extraction completed!")
    print(f"Total videos: {len(df)}")
    print(f"CSV saved to: {output_csv}")
    print(f"NPY files in: {output_dir}")


if __name__ == "__main__":
    extract_all_videos()
