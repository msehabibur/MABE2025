import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_v1 import (
    add_curvature_features,
    add_multiscale_features,
    add_state_features,
    add_longrange_features,
    add_interaction_features,
    transform_single,
    transform_pair
)


@pytest.fixture
def simple_trajectory():
    """Create a simple trajectory for testing"""
    n = 100
    t = np.linspace(0, 4 * np.pi, n)
    x = pd.Series(10 * np.cos(t) + 50)
    y = pd.Series(10 * np.sin(t) + 50)
    return x, y


@pytest.fixture
def mouse_single_data():
    """Create single mouse data for testing"""
    n_frames = 100
    body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

    data = {}
    for part in body_parts:
        data[(part, 'x')] = np.random.randn(n_frames) * 5 + 50
        data[(part, 'y')] = np.random.randn(n_frames) * 5 + 50

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def mouse_pair_data():
    """Create mouse pair data for testing"""
    n_frames = 100
    body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

    data = {}
    for mouse in ['A', 'B']:
        for part in body_parts:
            offset = 50 if mouse == 'A' else 60
            data[(mouse, part, 'x')] = np.random.randn(n_frames) * 5 + offset
            data[(mouse, part, 'y')] = np.random.randn(n_frames) * 5 + offset

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


class TestCurvatureFeatures:
    """Test curvature feature extraction"""

    def test_add_curvature_features_basic(self, simple_trajectory):
        """Test add_curvature_features creates features"""
        x, y = simple_trajectory
        X = pd.DataFrame()
        fps = 30.0

        result = add_curvature_features(X, x, y, fps)

        assert 'curv_mean_30' in result.columns
        assert 'curv_mean_60' in result.columns
        assert 'turn_rate_30' in result.columns
        assert len(result) == len(x)

    def test_add_curvature_features_different_fps(self, simple_trajectory):
        """Test curvature features with different fps"""
        x, y = simple_trajectory
        X = pd.DataFrame()

        result_30 = add_curvature_features(X.copy(), x, y, fps=30.0)
        result_60 = add_curvature_features(X.copy(), x, y, fps=60.0)

        assert len(result_30.columns) == len(result_60.columns)


class TestMultiscaleFeatures:
    """Test multiscale feature extraction"""

    def test_add_multiscale_features_basic(self, simple_trajectory):
        """Test add_multiscale_features creates features"""
        x, y = simple_trajectory
        X = pd.DataFrame()
        fps = 30.0

        result = add_multiscale_features(X, x, y, fps)

        assert 'sp_m10' in result.columns
        assert 'sp_m40' in result.columns
        assert 'sp_m160' in result.columns
        assert 'sp_s10' in result.columns
        assert 'sp_ratio' in result.columns

    def test_add_multiscale_features_short_series(self):
        """Test multiscale features with short time series"""
        x = pd.Series(np.random.randn(20))
        y = pd.Series(np.random.randn(20))
        X = pd.DataFrame()

        result = add_multiscale_features(X, x, y, fps=30.0)
        assert len(result) == len(x)


class TestStateFeatures:
    """Test behavioral state feature extraction"""

    def test_add_state_features_basic(self, simple_trajectory):
        """Test add_state_features creates features"""
        x, y = simple_trajectory
        X = pd.DataFrame()
        fps = 30.0

        result = add_state_features(X, x, y, fps)

        # Check for state features
        assert any('s0_' in col for col in result.columns) or len(result.columns) == 0

    def test_add_state_features_different_fps(self, simple_trajectory):
        """Test state features scale with fps"""
        x, y = simple_trajectory
        X = pd.DataFrame()

        result_30 = add_state_features(X.copy(), x, y, fps=30.0)
        result_60 = add_state_features(X.copy(), x, y, fps=60.0)

        # Both should produce features
        assert len(result_30) == len(x)
        assert len(result_60) == len(x)


class TestLongrangeFeatures:
    """Test long-range temporal feature extraction"""

    def test_add_longrange_features_basic(self, simple_trajectory):
        """Test add_longrange_features creates features"""
        x, y = simple_trajectory
        X = pd.DataFrame()
        fps = 30.0

        result = add_longrange_features(X, x, y, fps)

        assert 'x_ml120' in result.columns
        assert 'y_ml120' in result.columns
        assert 'x_e60' in result.columns
        assert 'y_e60' in result.columns

    def test_add_longrange_features_short_series(self):
        """Test longrange features with short series"""
        x = pd.Series(np.random.randn(50))
        y = pd.Series(np.random.randn(50))
        X = pd.DataFrame()

        result = add_longrange_features(X, x, y, fps=30.0)
        assert len(result) == len(x)


class TestInteractionFeatures:
    """Test social interaction feature extraction"""

    def test_add_interaction_features_basic(self, mouse_pair_data):
        """Test add_interaction_features creates features"""
        X = pd.DataFrame(index=range(len(mouse_pair_data)))
        fps = 30.0

        avail_A = mouse_pair_data['A'].columns.get_level_values(0)
        avail_B = mouse_pair_data['B'].columns.get_level_values(0)

        result = add_interaction_features(X, mouse_pair_data, avail_A, avail_B, fps)

        assert 'A_ld30' in result.columns
        assert 'B_ld30' in result.columns
        assert 'chase_30' in result.columns

    def test_add_interaction_features_missing_parts(self):
        """Test interaction features with missing body parts"""
        n = 100
        data = {}
        for mouse in ['A', 'B']:
            data[(mouse, 'nose', 'x')] = np.random.randn(n)
            data[(mouse, 'nose', 'y')] = np.random.randn(n)

        mouse_pair = pd.DataFrame(data)
        mouse_pair.columns = pd.MultiIndex.from_tuples(mouse_pair.columns)

        X = pd.DataFrame(index=range(len(mouse_pair)))
        avail_A = mouse_pair['A'].columns.get_level_values(0)
        avail_B = mouse_pair['B'].columns.get_level_values(0)

        result = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps=30.0)
        # Should return X unchanged if body_center is missing
        assert len(result) == len(X)


class TestTransformSingle:
    """Test single mouse transformation"""

    def test_transform_single_basic(self, mouse_single_data):
        """Test transform_single creates features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        fps = 30.0

        result = transform_single(mouse_single_data, body_parts, fps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mouse_single_data)
        assert result.shape[1] > 0
        assert result.dtype == np.float32

    def test_transform_single_different_fps(self, mouse_single_data):
        """Test transform_single with different fps values"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

        result_30 = transform_single(mouse_single_data, body_parts, fps=30.0)
        result_60 = transform_single(mouse_single_data, body_parts, fps=60.0)

        # Both should produce the same number of rows
        assert len(result_30) == len(result_60)
        # And same columns (same features)
        assert set(result_30.columns) == set(result_60.columns)

    def test_transform_single_distance_features(self, mouse_single_data):
        """Test transform_single creates distance features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_single(mouse_single_data, body_parts, fps=30.0)

        # Check for distance features
        assert 'nose+body_center' in result.columns
        assert 'nose+tail_base' in result.columns

    def test_transform_single_temporal_features(self, mouse_single_data):
        """Test transform_single creates temporal features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_single(mouse_single_data, body_parts, fps=30.0)

        # Check for temporal features
        assert any('cx_m' in col for col in result.columns)
        assert any('cy_m' in col for col in result.columns)

    def test_transform_single_elongation_feature(self, mouse_single_data):
        """Test transform_single creates elongation feature"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_single(mouse_single_data, body_parts, fps=30.0)

        assert 'elong' in result.columns

    def test_transform_single_body_angle(self, mouse_single_data):
        """Test transform_single creates body angle feature"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_single(mouse_single_data, body_parts, fps=30.0)

        assert 'body_ang' in result.columns

    def test_transform_single_curvature(self, mouse_single_data):
        """Test transform_single includes curvature features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_single(mouse_single_data, body_parts, fps=30.0)

        assert any('curv_mean' in col for col in result.columns)

    def test_transform_single_minimal_parts(self):
        """Test transform_single with minimal body parts"""
        n = 100
        data = {
            ('nose', 'x'): np.random.randn(n),
            ('nose', 'y'): np.random.randn(n),
            ('tail_base', 'x'): np.random.randn(n),
            ('tail_base', 'y'): np.random.randn(n),
        }
        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        body_parts = ['nose', 'tail_base']
        result = transform_single(df, body_parts, fps=30.0)

        assert len(result) == len(df)
        assert 'nose+tail_base' in result.columns


class TestTransformPair:
    """Test mouse pair transformation"""

    def test_transform_pair_basic(self, mouse_pair_data):
        """Test transform_pair creates features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        fps = 30.0

        result = transform_pair(mouse_pair_data, body_parts, fps)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(mouse_pair_data)
        assert result.shape[1] > 0
        assert result.dtype == np.float32

    def test_transform_pair_different_fps(self, mouse_pair_data):
        """Test transform_pair with different fps values"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

        result_30 = transform_pair(mouse_pair_data, body_parts, fps=30.0)
        result_60 = transform_pair(mouse_pair_data, body_parts, fps=60.0)

        assert len(result_30) == len(result_60)
        assert set(result_30.columns) == set(result_60.columns)

    def test_transform_pair_distance_features(self, mouse_pair_data):
        """Test transform_pair creates inter-mouse distance features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        # Check for inter-mouse distance features
        assert any('12+' in col for col in result.columns)

    def test_transform_pair_distance_bins(self, mouse_pair_data):
        """Test transform_pair creates distance bin features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        assert 'v_cls' in result.columns
        assert 'cls' in result.columns
        assert 'med' in result.columns
        assert 'far' in result.columns

    def test_transform_pair_relative_orientation(self, mouse_pair_data):
        """Test transform_pair creates relative orientation feature"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        assert 'rel_ori' in result.columns

    def test_transform_pair_approach_rate(self, mouse_pair_data):
        """Test transform_pair creates approach rate feature"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        assert 'appr' in result.columns

    def test_transform_pair_temporal_features(self, mouse_pair_data):
        """Test transform_pair creates temporal interaction features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        # Check for temporal features
        assert any('d_m' in col for col in result.columns)
        assert any('d_s' in col for col in result.columns)

    def test_transform_pair_coordination_features(self, mouse_pair_data):
        """Test transform_pair creates coordination features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        # Check for coordination features
        assert any('co_m' in col for col in result.columns)

    def test_transform_pair_velocity_alignment(self, mouse_pair_data):
        """Test transform_pair creates velocity alignment features"""
        body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']
        result = transform_pair(mouse_pair_data, body_parts, fps=30.0)

        # Check for velocity alignment features
        assert any('va_' in col for col in result.columns)

    def test_transform_pair_minimal_parts(self):
        """Test transform_pair with minimal body parts"""
        n = 100
        data = {}
        for mouse in ['A', 'B']:
            data[(mouse, 'nose', 'x')] = np.random.randn(n)
            data[(mouse, 'nose', 'y')] = np.random.randn(n)
            data[(mouse, 'tail_base', 'x')] = np.random.randn(n)
            data[(mouse, 'tail_base', 'y')] = np.random.randn(n)

        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        body_parts = ['nose', 'tail_base']
        result = transform_pair(df, body_parts, fps=30.0)

        assert len(result) == len(df)
        assert any('12+' in col for col in result.columns)
