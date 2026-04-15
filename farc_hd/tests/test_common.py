import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from farc_hd.FarcHDClassifier import FarcHDClassifier


# Instantiate the model with minimal resources to prevent 
# the default tests from melting GitHub Actions' RAM.
lightweight_model = FarcHDClassifier(
    n_labels=3,
    depth=2,
    max_trials=20,
    population_size=5
)

@parametrize_with_checks([lightweight_model])
def test_estimators(estimator, check, request):
    """Check the compatibility with scikit-learn API."""
    check(estimator)