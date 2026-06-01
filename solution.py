# compute_pca.py
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA, IncrementalPCA
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = "dataset/train"
IMG_SIZE = (150, 150)  # matches assignment
CLASSES = sorted([d.name for d in Path(DATA_DIR).iterdir() if d.is_dir()])

def load_images(max_per_class=None):
    print('Inside load images function ***')
    X, y, paths = [], [], []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = Path(DATA_DIR) / cls
        files = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if max_per_class:
            files = files[:max_per_class]
        for p in files:
            img = Image.open(p).convert("RGB").resize(IMG_SIZE)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            X.append(arr.ravel())
            y.append(cls_idx)
            paths.append(str(p))
    print('Returning from load images function ***')
    return np.stack(X), np.array(y), paths

if __name__ == "__main__":
    print('Executing the PCA analysis script ***')
    X, y, paths = load_images()
    # mean-center(standardization of data)
    X_mean = X.mean(axis=0)
    Xc = X - X_mean

    # PCA for visualization
    print('Perform PCA **')
    pca = PCA(n_components=2, svd_solver="randomized", random_state=0)
    Z = pca.fit_transform(Xc)
    print("Explained variance (2 PCs):", pca.explained_variance_ratio_.sum())

    print('Plotting PCA scatter plot ***')
    # scatter
    plt.figure(figsize=(8,6))
    for cls_idx, cls in enumerate(CLASSES):
        mask = y==cls_idx
        plt.scatter(Z[mask,0], Z[mask,1], s=8, label=cls, alpha=0.6)
    plt.legend(markerscale=2)
    plt.title("PCA (2D) of images (raw pixels)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()

    print('Identifying outliers based on PCA projection ***')

    # Show top outliers by projection distance from class centroid
    outlier_scores = []
    for cls_idx in range(len(CLASSES)):
        mask = (y==cls_idx)
        centroid = Z[mask].mean(axis=0)
        dists = np.linalg.norm(Z[mask] - centroid, axis=1)
        # store global mapping
        idxs = np.where(mask)[0]
        for i, dist in zip(idxs, dists):
            outlier_scores.append((dist, i))
    outlier_scores.sort(reverse=True)
    top = outlier_scores[:50]
    # Save CSV for review
    import csv
    print('Writing results to analysis/pca_outliers.csv ***')
    with open("analysis/pca_outliers.csv","w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["score","index","path","label"])
        for score,i in top:
            writer.writerow([score, i, paths[i], CLASSES[y[i]]])
    print("Wrote analysis/pca_outliers.csv (top suspicious samples)")