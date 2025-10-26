import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_v1 import _scale, _scale_signed, _fps_from_meta, safe_rolling


class TestFPSScalingFunctions:
    """Test FPS scaling helper functions"""

    def test_scale_basic(self):
        """Test basic _scale function"""
        # 30 frames at 30 fps should stay 30
        result = _scale(30, 30.0, ref=30.0)
        assert result == 30

    def test_scale_higher_fps(self):
        """Test _scale with higher fps"""
        # 30 frames at 60 fps should be 60 (double)
        result = _scale(30, 60.0, ref=30.0)
        assert result == 60

    def test_scale_lower_fps(self):
        """Test _scale with lower fps"""
        # 30 frames at 15 fps should be 15 (half)
        result = _scale(30, 15.0, ref=30.0)
        assert result == 15

    def test_scale_minimum_value(self):
        """Test _scale returns at least 1"""
        # Even with very low fps, should return at least 1
        result = _scale(10, 1.0, ref=30.0)
        assert result >= 1

    def test_scale_zero_frames(self):
        """Test _scale with zero frames"""
        result = _scale(0, 30.0, ref=30.0)
        assert result >= 1

    def test_scale_different_reference(self):
        """Test _scale with different reference fps"""
        result = _scale(60, 30.0, ref=60.0)
        assert result == 30

    def test_scale_signed_positive(self):
        """Test _scale_signed with positive value"""
        result = _scale_signed(30, 30.0, ref=30.0)
        assert result == 30

    def test_scale_signed_negative(self):
        """Test _scale_signed with negative value"""
        result = _scale_signed(-30, 30.0, ref=30.0)
        assert result == -30

    def test_scale_signed_zero(self):
        """Test _scale_signed with zero"""
        result = _scale_signed(0, 30.0, ref=30.0)
        assert result == 0

    def test_scale_signed_positive_higher_fps(self):
        """Test _scale_signed with positive value and higher fps"""
        result = _scale_signed(30, 60.0, ref=30.0)
        assert result == 60

    def test_scale_signed_negative_higher_fps(self):
        """Test _scale_signed with negative value and higher fps"""
        result = _scale_signed(-30, 60.0, ref=30.0)
        assert result == -60

    def test_scale_signed_minimum_magnitude(self):
        """Test _scale_signed maintains minimum magnitude of 1"""
        result = _scale_signed(10, 1.0, ref=30.0)
        assert abs(result) >= 1

    def test_scale_signed_preserves_sign(self):
        """Test _scale_signed preserves sign"""
        pos_result = _scale_signed(20, 15.0, ref=30.0)
        neg_result = _scale_signed(-20, 15.0, ref=30.0)
        assert pos_result > 0
        assert neg_result < 0

    def test_fps_from_meta_with_column(self):
        """Test _fps_from_meta when fps column exists"""
        meta = pd.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'frames_per_second': [25.0, 25.0]
        })
        result = _fps_from_meta(meta, {}, default_fps=30.0)
        assert result == 25.0

    def test_fps_from_meta_from_lookup(self):
        """Test _fps_from_meta using lookup dict"""
        meta = pd.DataFrame({
            'video_id': ['vid1', 'vid1']
        })
        lookup = {'vid1': 20.0}
        result = _fps_from_meta(meta, lookup, default_fps=30.0)
        assert result == 20.0

    def test_fps_from_meta_default(self):
        """Test _fps_from_meta with default value"""
        meta = pd.DataFrame({
            'video_id': ['vid2', 'vid2']
        })
        lookup = {'vid1': 20.0}
        result = _fps_from_meta(meta, lookup, default_fps=30.0)
        assert result == 30.0

    def test_fps_from_meta_null_values(self):
        """Test _fps_from_meta with null fps values"""
        meta = pd.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'frames_per_second': [None, None]
        })
        lookup = {'vid1': 25.0}
        result = _fps_from_meta(meta, lookup, default_fps=30.0)
        assert result == 25.0

    def test_safe_rolling_basic(self):
        """Test safe_rolling with basic operation"""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = safe_rolling(series, window=3, func=np.mean)
        assert len(result) == len(series)
        assert not result.isna().all()

    def test_safe_rolling_min_periods(self):
        """Test safe_rolling with custom min_periods"""
        series = pd.Series([1, 2, 3, 4, 5])
        result = safe_rolling(series, window=5, func=np.mean, min_periods=2)
        assert len(result) == len(series)

    def test_safe_rolling_with_nan(self):
        """Test safe_rolling handles NaN values"""
        series = pd.Series([1, np.nan, 3, 4, 5])
        result = safe_rolling(series, window=3, func=np.nanmean)
        assert len(result) == len(series)
