import numpy as np
from align import align_embeddings

E0 = np.load("../phase1_runs/seed_0/final_embedding.npy")
E1 = np.load("../phase1_runs/seed_1/final_embedding.npy")

# Center: remove the per-column mean. This removes the "common direction"
# that all tokens share.
E0_c = E0 - E0.mean(axis=0, keepdims=True)
E1_c = E1 - E1.mean(axis=0, keepdims=True)
print("After centering, Frobenius norms:")
print(f"  E0_c: {np.linalg.norm(E0_c):.2f}")
print(f"  E1_c: {np.linalg.norm(E1_c):.2f}")

# Procrustes on uncentered.
_, rho_uncentered = align_embeddings(E0, E1)
print(f"Uncentered ρ_E: {rho_uncentered:.4f}")

# Procrustes on centered.
_, rho_centered = align_embeddings(E0_c, E1_c)
print(f"Centered   ρ_E: {rho_centered:.4f}")

# Also try: remove just the top singular direction from each.
U0, S0, Vt0 = np.linalg.svd(E0, full_matrices=False)
U1, S1, Vt1 = np.linalg.svd(E1, full_matrices=False)
E0_noTop = E0 - np.outer(U0[:, 0] * S0[0], Vt0[0, :])
E1_noTop = E1 - np.outer(U1[:, 0] * S1[0], Vt1[0, :])
print(f"After removing top SV, Frobenius norms:")
print(f"  E0_noTop: {np.linalg.norm(E0_noTop):.2f}")
print(f"  E1_noTop: {np.linalg.norm(E1_noTop):.2f}")
_, rho_noTop = align_embeddings(E0_noTop, E1_noTop)
print(f"Top-SV-removed ρ_E: {rho_noTop:.4f}")

# And try aligning ONLY the top singular direction.
# Project each row onto its top singular component.
e0_top = U0[:, 0:1] * S0[0]  # (V, 1) — but we want the (1, H) direction Vt0[0:1]
# Actually just compare the top-1 singular vector directly.
v0 = Vt0[0]  # (H,)
v1 = Vt1[0]  # (H,)
# Best rotation to align v0 to v1 is trivial: any rotation that sends v0
# to v1. We just want to know if they're parallel or anti-parallel modulo sign.
print(f"|<v0, v1>|: {abs(v0 @ v1):.4f}  (1.0 = parallel, 0 = orthogonal)")
