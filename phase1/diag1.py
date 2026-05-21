import numpy as np

E0 = np.load("../phase1_runs/seed_0/final_embedding.npy")
E1 = np.load("../phase1_runs/seed_1/final_embedding.npy")
print("E0 shape:", E0.shape)
print("E0 Frobenius norm:", np.linalg.norm(E0))
print("E1 Frobenius norm:", np.linalg.norm(E1))
print("Per-row norm: E0 mean", np.linalg.norm(E0, axis=1).mean(),
      "std", np.linalg.norm(E0, axis=1).std())
print("Per-row norm: E1 mean", np.linalg.norm(E1, axis=1).mean(),
      "std", np.linalg.norm(E1, axis=1).std())

# Are the rows wildly different in magnitude?
print("Max per-row norm:", np.linalg.norm(E0, axis=1).max())
print("Min per-row norm:", np.linalg.norm(E0, axis=1).min())

# What does the spectrum look like?
sv0 = np.linalg.svd(E0, compute_uv=False)
sv1 = np.linalg.svd(E1, compute_uv=False)
print("Top 10 singular values, seed 0:", sv0[:10])
print("Top 10 singular values, seed 1:", sv1[:10])
print("E0 rank-effective:", float(np.exp(-np.sum((sv0**2 / (sv0**2).sum()) * np.log(sv0**2 / (sv0**2).sum() + 1e-30)))))
