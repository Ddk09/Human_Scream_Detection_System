import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

scream_dir = "scream_dataset/scream"
non_scream_dir = "scream_dataset/non_scream"

output_csv = "features_augmented.csv"


def extract_features(path, n_mfcc=13):

    try:
        y, sr = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        print("Error loading file:", path, e)
        return None

    if len(y) < 5:
        return None

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    return np.hstack([mfcc_mean, zcr, centroid, rolloff])

rows = []

for folder, label in [(scream_dir, 1), (non_scream_dir, 0)]:
    print("Processing folder:", folder)

    for fname in tqdm(os.listdir(folder)):
        if not fname.lower().endswith(".wav"):
            continue

        fpath = os.path.join(folder, fname)
        feats = extract_features(fpath)

        if feats is None:
            continue

        rows.append(np.hstack([feats, label]))

col_names = [f"mfcc_{i+1}" for i in range(13)]
col_names += ["zcr", "centroid", "rolloff", "label"]

df = pd.DataFrame(rows, columns=col_names)

df.to_csv(output_csv, index=False)

print("\nFinished. Features written to:", output_csv)

