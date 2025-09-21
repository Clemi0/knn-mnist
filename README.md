# Handwritten Digit Classification with k-Nearest Neighbors

This project extends a course lab into a **standalone portfolio project**: a from-scratch implementation of **k-Nearest Neighbors (kNN)** for classifying handwritten digits on the MNIST subset. It uses **vectorized NumPy** (no scikit-learn kNN) and includes hyperparameter tuning, normalization, multiple distance metrics, and error analysis.

## What’s inside?
- `lab01_extended.ipynb`: end-to-end notebook (data download, preprocessing, kNN implementation, experiments, plots).
- From-scratch functions: `dist_single`, `dist_all`, `predict_knn`, `compute_accuracy`.
- Extras: **k=1–50 validation curve**, **L1/L2/Cosine** metrics, **confusion matrix**, **misclassified samples** visualization.
- Ready for **Google Colab** (or local Jupyter).

## Setup
**Option A — Google Colab (recommended)**
1. Open the notebook in Colab.
2. Run the first cell to download & unzip the MNIST subset.
3. Run all cells.

**Option B — Local**
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
jupyter notebook
```

## Reported Results (expected)
- Unnormalized: ~25–30% validation accuracy.
- Normalized: ~**90%** validation accuracy (best k ≈ 5–7).

> Results can vary slightly due to data shuffling and environment.

## Skills Demonstrated
Vectorized NumPy • Data preprocessing • Model evaluation • Hyperparameter tuning • Visualization • Git/GitHub workflow

## Next Steps
- Compare with `sklearn.neighbors.KNeighborsClassifier`.
- Speed up search with KD-Tree / Ball-Tree.
- Add a simple CNN baseline in PyTorch.

---

**Dataset credit:** MNIST subset via course link from Yann LeCun’s MNIST.
