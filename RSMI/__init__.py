"""
RSMI: Real-Space Mutual Information neural estimator for symmetry-aware
coarse-graining of lattice statistical-mechanics models (RSMI-NE).
"""
from .connected_correlation import (
    run_connected_correlation,
    scaling_dimensions_from_correlations,
    nearest_neighbors_hamming_one,
)
from .training import (
    CoarseGrainer,
    SeparableCritic,
    train_RSMI_optimiser,
    mlp,
    infonce_lower_bound,
    lowerbounds,
    TimeHistory,
)
from .symmetry import (
    identity_permutation,
    hyperoctahedral_permutations,
)
from .analysis import (
    augment_by_permutations,
    multivariate_fit,
)
from .plotting import (
    plot_circle_graph,
    DrawCircle,
    build_segments_and_weights,
)
from .ve_dataset import (
    build_ve_indices,
    samples_to_ve_dataset,
)
from .tfrec_dataset import build_tfrecord_dataset

__all__ = [
    "CoarseGrainer",
    "SeparableCritic",
    "train_RSMI_optimiser",
    "mlp",
    "infonce_lower_bound",
    "lowerbounds",
    "TimeHistory",
    "identity_permutation",
    "hyperoctahedral_permutations",
    "augment_by_permutations",
    "multivariate_fit",
    "plot_circle_graph",
    "DrawCircle",
    "build_segments_and_weights",
    "build_ve_indices",
    "samples_to_ve_dataset",
    "build_tfrecord_dataset",
    "run_connected_correlation",
    "scaling_dimensions_from_correlations",
    "nearest_neighbors_hamming_one",
]
