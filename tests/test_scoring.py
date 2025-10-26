import pytest
import numpy as np
import pandas as pd
import polars as pl
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_v1 import single_lab_f1, mouse_fbeta, score, HostVisibleError


class TestScoringFunctions:
    """Test scoring functions"""

    def test_single_lab_f1_perfect_match(self):
        """Test single_lab_f1 with perfect prediction"""
        solution = pl.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'label_key': ['vid1_1_1_rear', 'vid1_1_1_walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80],
            'behaviors_labeled': ['["mouse1,self,rear","mouse1,self,walk"]'] * 2
        })

        submission = pl.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'agent_id': ['mouse1', 'mouse1'],
            'target_id': ['self', 'self'],
            'action': ['rear', 'walk'],
            'prediction_key': ['vid1_1_1_rear', 'vid1_1_1_walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80]
        })

        score_val = single_lab_f1(solution, submission, beta=1.0)
        assert score_val >= 0.9  # Should be very high for perfect match

    def test_single_lab_f1_partial_match(self):
        """Test single_lab_f1 with partial overlap"""
        solution = pl.DataFrame({
            'video_id': ['vid1'],
            'label_key': ['vid1_1_1_rear'],
            'start_frame': [0],
            'stop_frame': [30],
            'behaviors_labeled': ['["mouse1,self,rear"]']
        })

        submission = pl.DataFrame({
            'video_id': ['vid1'],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'prediction_key': ['vid1_1_1_rear'],
            'start_frame': [10],
            'stop_frame': [25]
        })

        score_val = single_lab_f1(solution, submission, beta=1.0)
        assert 0 < score_val < 1.0

    def test_single_lab_f1_no_overlap(self):
        """Test single_lab_f1 with no overlap"""
        solution = pl.DataFrame({
            'video_id': ['vid1'],
            'label_key': ['vid1_1_1_rear'],
            'start_frame': [0],
            'stop_frame': [30],
            'behaviors_labeled': ['["mouse1,self,rear"]']
        })

        submission = pl.DataFrame({
            'video_id': ['vid1'],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'prediction_key': ['vid1_1_1_rear'],
            'start_frame': [50],
            'stop_frame': [80]
        })

        score_val = single_lab_f1(solution, submission, beta=1.0)
        assert score_val == 0.0

    def test_single_lab_f1_duplicate_frames_error(self):
        """Test single_lab_f1 raises error on duplicate predictions"""
        solution = pl.DataFrame({
            'video_id': ['vid1'],
            'label_key': ['vid1_1_1_rear'],
            'start_frame': [0],
            'stop_frame': [30],
            'behaviors_labeled': ['["mouse1,self,rear"]']
        })

        # Two overlapping predictions from same agent/target pair
        submission = pl.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'agent_id': ['mouse1', 'mouse1'],
            'target_id': ['self', 'self'],
            'action': ['rear', 'rear'],
            'prediction_key': ['vid1_1_1_rear', 'vid1_1_1_rear'],
            'start_frame': [0, 10],
            'stop_frame': [20, 30]
        })

        with pytest.raises(HostVisibleError):
            single_lab_f1(solution, submission, beta=1.0)

    def test_single_lab_f1_missing_predictions(self):
        """Test single_lab_f1 with missing predictions"""
        solution = pl.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'label_key': ['vid1_1_1_rear', 'vid1_1_1_walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80],
            'behaviors_labeled': ['["mouse1,self,rear","mouse1,self,walk"]'] * 2
        })

        # Only predict one action
        submission = pl.DataFrame({
            'video_id': ['vid1'],
            'agent_id': ['mouse1'],
            'target_id': ['self'],
            'action': ['rear'],
            'prediction_key': ['vid1_1_1_rear'],
            'start_frame': [0],
            'stop_frame': [30]
        })

        score_val = single_lab_f1(solution, submission, beta=1.0)
        assert 0 < score_val < 1.0

    def test_mouse_fbeta_valid_input(self):
        """Test mouse_fbeta with valid input"""
        solution = pd.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'lab_id': ['lab1', 'lab1'],
            'agent_id': [1, 1],
            'target_id': [1, 1],
            'action': ['rear', 'walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80],
            'behaviors_labeled': ['["mouse1,self,rear","mouse1,self,walk"]'] * 2
        })

        submission = pd.DataFrame({
            'video_id': ['vid1', 'vid1'],
            'agent_id': [1, 1],
            'target_id': [1, 1],
            'action': ['rear', 'walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80]
        })

        score_val = mouse_fbeta(solution, submission, beta=1.0)
        assert isinstance(score_val, float)
        assert score_val >= 0.0

    def test_mouse_fbeta_empty_solution(self):
        """Test mouse_fbeta with empty solution"""
        solution = pd.DataFrame({
            'video_id': [],
            'lab_id': [],
            'agent_id': [],
            'target_id': [],
            'action': [],
            'start_frame': [],
            'stop_frame': [],
            'behaviors_labeled': []
        })

        submission = pd.DataFrame({
            'video_id': ['vid1'],
            'agent_id': [1],
            'target_id': [1],
            'action': ['rear'],
            'start_frame': [0],
            'stop_frame': [30]
        })

        with pytest.raises(ValueError):
            mouse_fbeta(solution, submission, beta=1.0)

    def test_mouse_fbeta_empty_submission(self):
        """Test mouse_fbeta with empty submission"""
        solution = pd.DataFrame({
            'video_id': ['vid1'],
            'lab_id': ['lab1'],
            'agent_id': [1],
            'target_id': [1],
            'action': ['rear'],
            'start_frame': [0],
            'stop_frame': [30],
            'behaviors_labeled': ['["mouse1,self,rear"]']
        })

        submission = pd.DataFrame({
            'video_id': [],
            'agent_id': [],
            'target_id': [],
            'action': [],
            'start_frame': [],
            'stop_frame': []
        })

        with pytest.raises(ValueError):
            mouse_fbeta(solution, submission, beta=1.0)

    def test_mouse_fbeta_missing_columns(self):
        """Test mouse_fbeta with missing columns"""
        solution = pd.DataFrame({
            'video_id': ['vid1'],
            'lab_id': ['lab1'],
            'agent_id': [1],
            'target_id': [1],
            'start_frame': [0],
            'stop_frame': [30],
            'behaviors_labeled': ['["mouse1,self,rear"]']
        })

        submission = pd.DataFrame({
            'video_id': ['vid1'],
            'agent_id': [1],
            'target_id': [1],
            'action': ['rear'],
            'start_frame': [0],
            'stop_frame': [30]
        })

        with pytest.raises(ValueError):
            mouse_fbeta(solution, submission, beta=1.0)

    def test_mouse_fbeta_multiple_labs(self):
        """Test mouse_fbeta with multiple labs"""
        solution = pd.DataFrame({
            'video_id': ['vid1', 'vid2'],
            'lab_id': ['lab1', 'lab2'],
            'agent_id': [1, 2],
            'target_id': [1, 2],
            'action': ['rear', 'walk'],
            'start_frame': [0, 0],
            'stop_frame': [30, 30],
            'behaviors_labeled': ['["mouse1,self,rear"]', '["mouse2,self,walk"]']
        })

        submission = pd.DataFrame({
            'video_id': ['vid1', 'vid2'],
            'agent_id': [1, 2],
            'target_id': [1, 2],
            'action': ['rear', 'walk'],
            'start_frame': [0, 0],
            'stop_frame': [30, 30]
        })

        score_val = mouse_fbeta(solution, submission, beta=1.0)
        assert isinstance(score_val, float)
        assert score_val >= 0.0

    def test_score_function(self):
        """Test score wrapper function"""
        solution = pd.DataFrame({
            'row_id': [0, 1],
            'video_id': ['vid1', 'vid1'],
            'lab_id': ['lab1', 'lab1'],
            'agent_id': [1, 1],
            'target_id': [1, 1],
            'action': ['rear', 'walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80],
            'behaviors_labeled': ['["mouse1,self,rear","mouse1,self,walk"]'] * 2
        })

        submission = pd.DataFrame({
            'row_id': [0, 1],
            'video_id': ['vid1', 'vid1'],
            'agent_id': [1, 1],
            'target_id': [1, 1],
            'action': ['rear', 'walk'],
            'start_frame': [0, 50],
            'stop_frame': [30, 80]
        })

        score_val = score(solution, submission, 'row_id', beta=1.0)
        assert isinstance(score_val, float)
        assert score_val >= 0.0
