"""Fusion methods for combining multiple retrieval results."""

from functools import partial

from .base import BaseFusion
from .interleaving import InterleavingFusion, NestedInterleavingFusion
from .rrf import ReciprocalRankFusion
from .concatenation import SimpleConcatenation
from .random_fusion import RandomFusion
from .epsilon_greedy import EpsilonGreedyFusion
from .thompson_bernoulli import ThompsonBernoulliFusion
from .thompson_gaussian import ThompsonGaussianFusion

# Backward-compatible aliases — these replace the old per-variant files.
# They behave identically to the former dedicated classes.
BernoulliUCBFusion = partial(ThompsonBernoulliFusion, ucb=True)
BernoulliTopKFusion = partial(ThompsonBernoulliFusion, k=5)
DiversityLinearFusion = partial(ThompsonBernoulliFusion, diversity="linear")
DiversityConcaveFusion = partial(ThompsonBernoulliFusion, diversity="concave")
BernoulliTopKUCBDiversityFusion = partial(ThompsonBernoulliFusion, k=5, ucb=True)
ThompsonGaussianNIGFusion = partial(ThompsonGaussianFusion, nig=True)

__all__ = [
    "BaseFusion",
    "InterleavingFusion",
    "NestedInterleavingFusion",
    "ReciprocalRankFusion",
    "SimpleConcatenation",
    "RandomFusion",
    "EpsilonGreedyFusion",
    "ThompsonBernoulliFusion",
    "ThompsonGaussianFusion",
    # Backward-compatible aliases
    "BernoulliUCBFusion",
    "BernoulliTopKFusion",
    "DiversityLinearFusion",
    "DiversityConcaveFusion",
    "BernoulliTopKUCBDiversityFusion",
    "ThompsonGaussianNIGFusion",
]
