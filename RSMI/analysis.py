"""
RSMI analysis: orbits, invariant polynomial fit, augmentation, spin importance, Metropolis optimization.
"""

import re
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

def augment_by_permutations(V, permutations):
    """V (N,D), permutations (K,D). Returns (K*N, D)."""
    perms = tf.convert_to_tensor(permutations, dtype=tf.int32)
    gathered = tf.gather(params=V, indices=perms, axis=1)
    return tf.reshape(tf.transpose(gathered, perm=[1, 0, 2]), (-1, tf.shape(V)[1]))


def multivariate_fit(x, y, d, interaction_only=False):
    """Polynomial regression without train/test split. Returns model, poly."""
    poly = PolynomialFeatures(
        degree=d, interaction_only=interaction_only, include_bias=False
    )
    Xp = poly.fit_transform(x)
    model = LinearRegression().fit(Xp, y)
    r2 = model.score(Xp, y)
    print(
        f"Polynomial surrogate fit: degree={d}, "
        f"interaction_only={interaction_only}, n_features={Xp.shape[1]} "
        f"-> R^2 = {r2:.4f}"
    )
    return model, poly