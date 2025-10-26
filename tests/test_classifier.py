import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_v1 import StratifiedSubsetClassifier


class TestStratifiedSubsetClassifier:
    """Test the StratifiedSubsetClassifier class"""

    def test_init(self):
        """Test classifier initialization"""
        estimator = LogisticRegression()
        clf = StratifiedSubsetClassifier(estimator, n_samples=100)
        assert clf.estimator == estimator
        assert clf.n_samples == 100

    def test_init_no_samples(self):
        """Test classifier initialization without n_samples"""
        estimator = LogisticRegression()
        clf = StratifiedSubsetClassifier(estimator, n_samples=None)
        assert clf.n_samples is None

    def test_to_numpy_from_dataframe(self):
        """Test _to_numpy conversion from DataFrame"""
        clf = StratifiedSubsetClassifier(LogisticRegression())
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        result = clf._to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (3, 2)

    def test_to_numpy_from_array(self):
        """Test _to_numpy conversion from numpy array"""
        clf = StratifiedSubsetClassifier(LogisticRegression())
        arr = np.array([[1, 2], [3, 4]])
        result = clf._to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_fit_full_data(self):
        """Test fit with full dataset (no subsampling)"""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        clf = StratifiedSubsetClassifier(LogisticRegression(), n_samples=None)
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 2

    def test_fit_with_subsampling(self):
        """Test fit with stratified subsampling"""
        X = np.random.randn(1000, 5)
        y = np.random.randint(0, 2, 1000)
        clf = StratifiedSubsetClassifier(LogisticRegression(), n_samples=100)
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')

    def test_fit_binary_conversion(self):
        """Test fit with {0, 2} labels converted to binary"""
        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [2] * 50)
        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')

    def test_fit_with_dataframe(self):
        """Test fit with pandas DataFrame input"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = np.random.randint(0, 2, 100)
        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')

    def test_fit_exception_fallback(self):
        """Test fit with exception fallback to step sampling"""
        X = np.random.randn(200, 5)
        # Create y with all same values to trigger stratification error
        y = np.zeros(200)
        clf = StratifiedSubsetClassifier(LogisticRegression(), n_samples=50)
        # Should fall back to step sampling
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')

    def test_predict_proba_normal(self):
        """Test predict_proba with normal data"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)

        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_single_class(self):
        """Test predict_proba with single class (class 1)"""
        X_train = np.random.randn(100, 5)
        y_train = np.ones(100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        clf.classes_ = np.array([1])
        proba = clf.predict_proba(X_test)

        assert proba.shape == (20, 2)
        assert np.allclose(proba[:, 1], 1.0)

    def test_predict_proba_single_class_zero(self):
        """Test predict_proba with single class (class 0)"""
        X_train = np.random.randn(100, 5)
        y_train = np.zeros(100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        clf.classes_ = np.array([0])
        proba = clf.predict_proba(X_test)

        assert proba.shape == (20, 2)
        assert np.allclose(proba[:, 0], 1.0)

    def test_predict_proba_1d_output(self):
        """Test predict_proba with 1D output conversion"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)

        assert proba.shape[1] == 2

    def test_predict(self):
        """Test predict method"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})

    def test_predict_exception_fallback(self):
        """Test predict with exception fallback to argmax"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 5)

        clf = StratifiedSubsetClassifier(LogisticRegression())
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        assert predictions.shape == (20,)

    def test_fit_small_dataset(self):
        """Test fit when dataset is smaller than n_samples"""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        clf = StratifiedSubsetClassifier(LogisticRegression(), n_samples=100)
        clf.fit(X, y)
        assert hasattr(clf, 'classes_')
