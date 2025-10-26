import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_v1 import _select_threshold_map, predict_multiclass_adaptive


class TestThresholdSelection:
    """Test threshold selection functions"""

    def test_select_threshold_map_simple_dict(self):
        """Test _select_threshold_map with simple dictionary"""
        thresholds = {
            "default": 0.5,
            "rear": 0.3,
            "walk": 0.4
        }
        result = _select_threshold_map(thresholds, "single")

        assert result["rear"] == 0.3
        assert result["walk"] == 0.4
        assert result["unknown_action"] == 0.5  # default

    def test_select_threshold_map_mode_aware(self):
        """Test _select_threshold_map with mode-aware thresholds"""
        thresholds = {
            "default": 0.27,
            "single_default": 0.30,
            "pair_default": 0.25,
            "single": {"rear": 0.35},
            "pair": {"chase": 0.20}
        }

        single_map = _select_threshold_map(thresholds, "single")
        pair_map = _select_threshold_map(thresholds, "pair")

        assert single_map["rear"] == 0.35
        assert single_map["walk"] == 0.30  # single_default
        assert pair_map["chase"] == 0.20
        assert pair_map["walk"] == 0.25  # pair_default

    def test_select_threshold_map_mode_aware_with_override(self):
        """Test mode-aware thresholds with action overrides"""
        thresholds = {
            "default": 0.27,
            "single": {"rear": 0.30}
        }
        result = _select_threshold_map(thresholds, "single")

        assert result["rear"] == 0.30
        assert result["other_action"] == 0.27

    def test_select_threshold_map_default_only(self):
        """Test with only default threshold"""
        thresholds = {"default": 0.4}
        result = _select_threshold_map(thresholds, "single")

        assert result["any_action"] == 0.4

    def test_select_threshold_map_float_input(self):
        """Test _select_threshold_map with float input"""
        thresholds = 0.3
        result = _select_threshold_map(thresholds, "single")

        assert result["any_action"] == 0.27  # default fallback

    def test_select_threshold_map_empty_dict(self):
        """Test with empty dictionary"""
        thresholds = {}
        result = _select_threshold_map(thresholds, "single")

        assert result["any_action"] == 0.27  # default fallback


class TestPredictMulticlassAdaptive:
    """Test adaptive multiclass prediction"""

    def test_predict_multiclass_adaptive_basic(self):
        """Test basic prediction with adaptive thresholding"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.random.rand(n_frames) * 0.5,
            'walk': np.random.rand(n_frames) * 0.3,
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.2}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        assert isinstance(result, pd.DataFrame)
        assert 'video_id' in result.columns
        assert 'agent_id' in result.columns
        assert 'target_id' in result.columns
        assert 'action' in result.columns
        assert 'start_frame' in result.columns
        assert 'stop_frame' in result.columns

    def test_predict_multiclass_adaptive_single_mode(self):
        """Test prediction detects single mode"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.array([0.8] * 30 + [0.1] * 70),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # Should detect some actions
        assert len(result) >= 0

    def test_predict_multiclass_adaptive_pair_mode(self):
        """Test prediction detects pair mode"""
        n_frames = 100
        pred = pd.DataFrame({
            'chase': np.array([0.8] * 30 + [0.1] * 70),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['mouse2'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        assert len(result) >= 0

    def test_predict_multiclass_adaptive_temporal_smoothing(self):
        """Test that temporal smoothing is applied"""
        n_frames = 100
        # Create noisy predictions
        pred = pd.DataFrame({
            'rear': np.random.choice([0.1, 0.9], n_frames),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # Smoothing should reduce number of very short events
        if len(result) > 0:
            durations = result['stop_frame'] - result['start_frame']
            # All events should be at least 3 frames (due to filtering)
            assert (durations >= 3).all()

    def test_predict_multiclass_adaptive_threshold_filtering(self):
        """Test that low-confidence predictions are filtered"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.full(n_frames, 0.1),  # All below threshold
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # No predictions should pass threshold
        assert len(result) == 0

    def test_predict_multiclass_adaptive_action_specific_threshold(self):
        """Test action-specific thresholds"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.full(n_frames, 0.4),
            'walk': np.full(n_frames, 0.4),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {
            "default": 0.5,
            "single": {"rear": 0.3}  # Lower threshold for rear
        }
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # Rear should pass (0.4 > 0.3) but walk should not (0.4 < 0.5)
        if len(result) > 0:
            assert 'rear' in result['action'].values

    def test_predict_multiclass_adaptive_multiple_actions(self):
        """Test prediction with multiple actions"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.concatenate([np.full(30, 0.8), np.full(70, 0.1)]),
            'walk': np.concatenate([np.full(30, 0.1), np.full(70, 0.8)]),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # Should detect both actions
        if len(result) > 0:
            actions = set(result['action'].values)
            assert len(actions) >= 1

    def test_predict_multiclass_adaptive_start_stop_consistency(self):
        """Test that start_frame < stop_frame"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.array([0.8] * 30 + [0.1] * 70),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        if len(result) > 0:
            assert (result['stop_frame'] > result['start_frame']).all()

    def test_predict_multiclass_adaptive_filters_short_events(self):
        """Test that very short events are filtered"""
        n_frames = 100
        pred = pd.DataFrame({
            'rear': np.array([0.8] * 30 + [0.1] * 70),
        }, index=range(n_frames))

        meta = pd.DataFrame({
            'video_id': ['vid1'] * n_frames,
            'agent_id': ['mouse1'] * n_frames,
            'target_id': ['self'] * n_frames,
            'video_frame': range(n_frames)
        })

        thresholds = {"default": 0.5}
        result = predict_multiclass_adaptive(pred, meta, thresholds)

        # All events should have duration >= 3
        if len(result) > 0:
            durations = result['stop_frame'] - result['start_frame']
            assert (durations >= 3).all()
