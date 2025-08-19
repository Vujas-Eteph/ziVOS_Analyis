"""
My current problem is that I'M assuming independance between my variavles
"""

import numpy as np
import torch
from torch.distributions import Categorical, kl_divergence

# Define X and Y
X = np.array([0.1, 0.7, 0.2])
Y = np.array([0.1, 0.7, 0.2])

# Compute outer product to create joint distribution P(X,Y)
P_XY = np.outer(X, Y)

# Normalize P_XY to ensure it's a proper joint distribution
P_XY /= P_XY.sum()

# Marginals P(X) and P(Y)
P_X = np.sum(P_XY, axis=1)  # Sum over columns to get P(X)
P_Y = np.sum(P_XY, axis=0)  # Sum over rows to get P(Y)

# Compute KL divergence D_KL(P(X,Y) || P(X)P(Y)) using NumPy
D_KL = 0.0
for i in range(P_XY.shape[0]):
    for j in range(P_XY.shape[1]):
        if P_XY[i, j] > 0:
            P_X_P_Y = P_X[i] * P_Y[j]
            if P_X_P_Y > 0:
                D_KL += P_XY[i, j] * np.log(P_XY[i, j] / P_X_P_Y)

print("KL Divergence D_KL(P(X,Y) || P(X)P(Y)): {:.4f}".format(D_KL))

# Convert P_XY to a tensor
P_XY_tensor = torch.tensor(P_XY, dtype=torch.float32)

# Compute marginals
P_X = P_XY_tensor.sum(1)
P_Y = P_XY_tensor.sum(0)

# Compute the product of marginals and normalize
P_X_product_P_Y = torch.outer(P_X, P_Y)
P_X_product_P_Y /= P_X_product_P_Y.sum()

# Flatten the distributions to 1D and create Categorical distributions
dist_P_XY = Categorical(probs=P_XY_tensor.view(-1))
dist_P_X_product_P_Y = Categorical(probs=P_X_product_P_Y.view(-1))

# Compute KL divergence using PyTorch
kl_div = kl_divergence(dist_P_XY, dist_P_X_product_P_Y)

print(f"KL Divergence with PyTorch: {kl_div.item()}")
