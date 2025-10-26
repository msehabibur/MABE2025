import pytest
import pandas as pd
import numpy as np
import polars as pl
from collections import defaultdict


@pytest.fixture
def simple_mouse_data():
    """Create simple mouse tracking data for testing"""
    n_frames = 100
    body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

    data = {}
    for part in body_parts:
        data[(part, 'x')] = np.random.randn(n_frames) * 10 + 50
        data[(part, 'y')] = np.random.randn(n_frames) * 10 + 50

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def simple_mouse_pair_data():
    """Create simple mouse pair tracking data for testing"""
    n_frames = 100
    body_parts = ['nose', 'body_center', 'tail_base', 'ear_left', 'ear_right']

    data = {}
    for mouse in ['A', 'B']:
        for part in body_parts:
            data[(mouse, part, 'x')] = np.random.randn(n_frames) * 10 + (50 if mouse == 'A' else 60)
            data[(mouse, part, 'y')] = np.random.randn(n_frames) * 10 + (50 if mouse == 'A' else 60)

    df = pd.DataFrame(data)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


@pytest.fixture
def simple_meta_data():
    """Create simple metadata for testing"""
    n_frames = 100
    return pd.DataFrame({
        'video_id': ['test_video'] * n_frames,
        'agent_id': ['mouse1'] * n_frames,
        'target_id': ['self'] * n_frames,
        'video_frame': range(n_frames),
        'frames_per_second': [30.0] * n_frames
    })


@pytest.fixture
def simple_predictions():
    """Create simple prediction dataframe"""
    n_frames = 100
    return pd.DataFrame({
        'rear': np.random.rand(n_frames) * 0.5,
        'walk': np.random.rand(n_frames) * 0.3,
    }, index=range(n_frames))


@pytest.fixture
def sample_solution_df():
    """Create sample solution dataframe for scoring"""
    data = {
        'video_id': ['vid1', 'vid1', 'vid2'],
        'lab_id': ['lab1', 'lab1', 'lab2'],
        'agent_id': [1, 1, 2],
        'target_id': [1, 1, 2],
        'action': ['rear', 'walk', 'rear'],
        'start_frame': [0, 50, 0],
        'stop_frame': [30, 80, 40],
        'behaviors_labeled': ['["mouse1,self,rear","mouse1,self,walk"]'] * 3
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_submission_df():
    """Create sample submission dataframe for scoring"""
    data = {
        'video_id': ['vid1', 'vid1', 'vid2'],
        'agent_id': [1, 1, 2],
        'target_id': [1, 1, 2],
        'action': ['rear', 'walk', 'rear'],
        'start_frame': [0, 45, 5],
        'stop_frame': [25, 75, 35]
    }
    return pd.DataFrame(data)
