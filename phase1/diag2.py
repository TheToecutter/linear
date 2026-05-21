# Procrustes alignment on the top-K most frequent tokens only.
# (Frequencies aren't saved but we can fake this by picking the K
# tokens with largest embedding norm in seed 0.)
import numpy as np
from align import align_embeddings

E0 = np.load("../phase1_runs/seed_0/final_embedding.npy")
E1 = np.load("../phase1_runs/seed_1/final_embedding.npy")

# Sort by per-row norm in E0, descending.
norms = np.linalg.norm(E0, axis=1)
top_indices = np.argsort(-norms)

for K in [10, 100, 1000, 5000, 32768]:
    idx = top_indices[:K]
    _, rho = align_embeddings(E0[idx], E1[idx])
    print(f"K = {K:>5}: ρ_E = {rho:.4f}")
