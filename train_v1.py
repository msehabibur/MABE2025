# Update of the Taylor S. Amarel and AmbrosM great notebook
# Removes the constant FPS assumption; handles variable frame timing
# The model underperforms on 'single' targets, so we train on more samples.
# Added action dependent adaptive threshold map

validate_or_submit = 'submit'
verbose = True

import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import warnings
import json
import os, random
import gc
import lightgbm
from collections import defaultdict
import polars as pl
from scipy import signal, stats

from sklearn.base import ClassifierMixin, BaseEstimator, clone
from sklearn.model_selection import cross_val_predict, GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

warnings.filterwarnings('ignore')

# Global containers for cross-validation reporting
cv_per_lab_reports = []
cv_macro_reports = []

# Try importing additional models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

# --- SEED EVERYTHING -----
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)      # has to be set very early

rnd = np.random.RandomState(SEED)
random.seed(SEED)
np.random.seed(SEED)
class StratifiedSubsetClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, n_samples=None):
        self.estimator = estimator
        self.n_samples = n_samples  # if None → no subsampling/stratification

    def _to_numpy(self, X):
        try:
            return X.to_numpy(np.float32, copy=False)
        except AttributeError:
            return np.asarray(X, dtype=np.float32)

    def fit(self, X, y):
        Xn = self._to_numpy(X)
        y = np.asarray(y).ravel()

        uniq = np.unique(y[~pd.isna(y)])
        if set(uniq.tolist()) == {0, 2}:
            y = (y > 0).astype(np.int8)

        # If n_samples is None → fit on full data, no stratification
        if self.n_samples is None or len(Xn) <= int(self.n_samples):
            self.estimator.fit(Xn, y)
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=int(self.n_samples), random_state=42)
            try:
                idx, _ = next(sss.split(np.zeros_like(y), y))
                self.estimator.fit(Xn[idx], y[idx])
            except Exception:
                step = max(len(Xn) // int(self.n_samples), 1)
                self.estimator.fit(Xn[::step], y[::step])

        try:
            self.classes_ = np.asarray(self.estimator.classes_)
        except Exception:
            self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        Xn = self._to_numpy(X)
        try:
            P = self.estimator.predict_proba(Xn)
        except Exception:
            if len(self.classes_) == 1:
                n = len(Xn)
                c = int(self.classes_[0])
                if c == 1:
                    return np.column_stack([np.zeros(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
                else:
                    return np.column_stack([np.ones(n, dtype=np.float32), np.zeros(n, dtype=np.float32)])
            return np.full((len(Xn), 2), 0.5, dtype=np.float32)

        P = np.asarray(P)
        if P.ndim == 1:
            P1 = P.astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        if P.shape[1] == 1 and len(self.classes_) == 2:
            P1 = P[:, 0].astype(np.float32)
            return np.column_stack([1.0 - P1, P1])
        return P

    def predict(self, X):
        Xn = self._to_numpy(X)
        try:
            return self.estimator.predict(Xn)
        except Exception:
            return np.argmax(self.predict_proba(Xn), axis=1)
# ==================== SCORING FUNCTIONS ====================

class HostVisibleError(Exception):
    pass

def single_lab_f1(lab_solution: pl.DataFrame, lab_submission: pl.DataFrame, beta: float = 1) -> float:
    label_frames: defaultdict[str, set[int]] = defaultdict(set)
    prediction_frames: defaultdict[str, set[int]] = defaultdict(set)

    for row in lab_solution.to_dicts():
        label_frames[row['label_key']].update(range(row['start_frame'], row['stop_frame']))

    for video in lab_solution['video_id'].unique():
        active_labels: str = lab_solution.filter(pl.col('video_id') == video)['behaviors_labeled'].first()
        active_labels: set[str] = set(json.loads(active_labels))
        predicted_mouse_pairs: defaultdict[str, set[int]] = defaultdict(set)

        for row in lab_submission.filter(pl.col('video_id') == video).to_dicts():
            if ','.join([str(row['agent_id']), str(row['target_id']), row['action']]) not in active_labels:
                continue
           
            new_frames = set(range(row['start_frame'], row['stop_frame']))
            new_frames = new_frames.difference(prediction_frames[row['prediction_key']])
            prediction_pair = ','.join([str(row['agent_id']), str(row['target_id'])])
            if predicted_mouse_pairs[prediction_pair].intersection(new_frames):
                raise HostVisibleError('Multiple predictions for the same frame from one agent/target pair')
            prediction_frames[row['prediction_key']].update(new_frames)
            predicted_mouse_pairs[prediction_pair].update(new_frames)

    tps = defaultdict(int)
    fns = defaultdict(int)
    fps = defaultdict(int)
    for key, pred_frames in prediction_frames.items():
        action = key.split('_')[-1]
        matched_label_frames = label_frames[key]
        tps[action] += len(pred_frames.intersection(matched_label_frames))
        fns[action] += len(matched_label_frames.difference(pred_frames))
        fps[action] += len(pred_frames.difference(matched_label_frames))

    distinct_actions = set()
    for key, frames in label_frames.items():
        action = key.split('_')[-1]
        distinct_actions.add(action)
        if key not in prediction_frames:
            fns[action] += len(frames)

    action_f1s = []
    for action in distinct_actions:
        if tps[action] + fns[action] + fps[action] == 0:
            action_f1s.append(0)
        else:
            action_f1s.append((1 + beta**2) * tps[action] / ((1 + beta**2) * tps[action] + beta**2 * fns[action] + fps[action]))
    return sum(action_f1s) / len(action_f1s)

def mouse_fbeta(solution: pd.DataFrame, submission: pd.DataFrame, beta: float = 1) -> float:
    if len(solution) == 0 or len(submission) == 0:
        raise ValueError('Missing solution or submission data')

    expected_cols = ['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame']

    for col in expected_cols:
        if col not in solution.columns:
            raise ValueError(f'Solution is missing column {col}')
        if col not in submission.columns:
            raise ValueError(f'Submission is missing column {col}')

    solution: pl.DataFrame = pl.DataFrame(solution)
    submission: pl.DataFrame = pl.DataFrame(submission)
    assert (solution['start_frame'] <= solution['stop_frame']).all()
    assert (submission['start_frame'] <= submission['stop_frame']).all()
    solution_videos = set(solution['video_id'].unique())
    submission = submission.filter(pl.col('video_id').is_in(solution_videos))

    solution = solution.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('label_key'),
    )
    submission = submission.with_columns(
        pl.concat_str(
            [
                pl.col('video_id').cast(pl.Utf8),
                pl.col('agent_id').cast(pl.Utf8),
                pl.col('target_id').cast(pl.Utf8),
                pl.col('action'),
            ],
            separator='_',
        ).alias('prediction_key'),
    )

    lab_scores = []
    for lab in solution['lab_id'].unique():
        lab_solution = solution.filter(pl.col('lab_id') == lab).clone()
        lab_videos = set(lab_solution['video_id'].unique())
        lab_submission = submission.filter(pl.col('video_id').is_in(lab_videos)).clone()
        lab_scores.append(single_lab_f1(lab_solution, lab_submission, beta=beta))

    return sum(lab_scores) / len(lab_scores)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, beta: float = 1) -> float:
    solution = solution.drop(row_id_column_name, axis='columns', errors='ignore')
    submission = submission.drop(row_id_column_name, axis='columns', errors='ignore')
    return mouse_fbeta(solution, submission, beta=beta)
# ==================== DATA LOADING ====================

train = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/train.csv')
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)

test = pd.read_csv('/kaggle/input/MABe-mouse-behavior-detection/test.csv')
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))

drop_body_parts = ['headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
                   'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
                   'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint']

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    assert traintest in ['train', 'test']
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
    for _, row in dataset.iterrows():
        
        lab_id = row.lab_id
        video_id = row.video_id

        if type(row.behaviors_labeled) != str:
            if verbose: print('No labeled behaviors:', lab_id, video_id)
            continue

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        if pvid.isna().any().any():
            if verbose and traintest == 'test': print('video with missing values', video_id, traintest, len(vid), 'frames')
        else:
            if verbose and traintest == 'test': print('video with all values', video_id, traintest, len(vid), 'frames')
        del vid
        pvid = pvid.reorder_levels([1, 2, 0], axis=1).T.sort_index().T
        pvid /= row.pix_per_cm_approx

        vid_behaviors = json.loads(row.behaviors_labeled)
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
        
        if traintest == 'train':
            try:
                annot = pd.read_parquet(path.replace('train_tracking', 'train_annotation'))
            except FileNotFoundError:
                continue

        if generate_single:
            vid_behaviors_subset = vid_behaviors.query("target == 'self'")
            for mouse_id_str in np.unique(vid_behaviors_subset.agent):
                try:
                    mouse_id = int(mouse_id_str[-1])
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("agent == @mouse_id_str").action)
                    single_mouse = pvid.loc[:, mouse_id]
                    assert len(single_mouse) == len(pvid)
                    single_mouse_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': mouse_id_str,
                        'target_id': 'self',
                        'video_frame': single_mouse.index,
                        'frames_per_second': row.frames_per_second,
                        'lab_id': lab_id,
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        if verbose: print('- test single', video_id, mouse_id)
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2):
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index,
                        'frames_per_second': row.frames_per_second,
                        'lab_id': lab_id,
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        if verbose: print('- test pair', video_id, agent, target)
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
# ==================== ADAPTIVE THRESHOLDING ====================

action_thresholds = {
    "default": 0.44,
    "single_default": 0.44,
    "pair_default": 0.44,
    "single": {
        "rear": 0.44,
    },
}

def _select_threshold_map(thresholds, mode: str):
    # same behavior you had, but returns a defaultdict
    if isinstance(thresholds, dict):
        # mode-aware?
        if ("single" in thresholds) or ("pair" in thresholds) or \
           ("single_default" in thresholds) or ("pair_default" in thresholds):
            base_default = float(thresholds.get("default", 0.27))
            mode_default = float(thresholds.get(f"{mode}_default", base_default))
            mode_overrides = thresholds.get(mode, {}) or {}
            out = defaultdict(lambda: mode_default)
            out.update({str(k): float(v) for k, v in mode_overrides.items()})
            return out
        # plain per-action dict
        out = defaultdict(lambda: float(thresholds.get("default", 0.27)))
        out.update({str(k): float(v) for k, v in thresholds.items() if k != "default"})
        return out
    return defaultdict(lambda: 0.27)

def predict_multiclass_adaptive(pred, meta, action_thresholds):
    """Adaptive thresholding per action + temporal smoothing"""
    # Apply temporal smoothing
    pred_smoothed = pred.rolling(window=5, min_periods=1, center=True).mean()


    mode = 'pair'
    try:
        if 'target_id' in meta.columns and meta['target_id'].eq('self').all():
            mode = 'single'
    except Exception:
        pass

    ama = np.argmax(pred_smoothed, axis=1)
    th_map = _select_threshold_map(action_thresholds, mode)

    max_probs = pred_smoothed.max(axis=1)
    threshold_mask = np.zeros(len(pred_smoothed), dtype=bool)
    for i, action in enumerate(pred_smoothed.columns):
        action_mask = (ama == i)
        threshold = th_map[action]
        threshold_mask |= (action_mask & (max_probs >= threshold))

    ama = np.where(threshold_mask, ama, -1)
    ama = pd.Series(ama, index=meta.video_frame)
    
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    mask = ama_changes.values >= 0
    mask[-1] = False
    
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        if i < len(stop_video_id):
            if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
                new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
                submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
        else:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame
    
    # Filter out very short events (likely noise)
    duration = submission_part.stop_frame - submission_part.start_frame
    submission_part = submission_part[duration >= 3].reset_index(drop=True)
    
    if len(submission_part) > 0:
        assert (submission_part.stop_frame > submission_part.start_frame).all(), 'stop <= start'
    
    if verbose: print(f'  actions found: {len(submission_part)}')
    return submission_part
# ==================== ADVANCED FEATURE ENGINEERING (FPS-AWARE) ====================

def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # invariant to time scaling

    for w in [30, 60]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    # displacement per frame is already in cm (pix normalized earlier); convert to cm/s
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions; bins adjusted so semantics are fps-invariant."""
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # Original bins (cm/frame): [-inf, 0.5, 2.0, 5.0, inf]
        # Convert to cm/s by multiplying by fps to keep thresholds consistent across fps.
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features (windows & spans scaled by fps)."""
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM spans also interpreted in frames
    for span in [60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

    # per-frame velocities (cm/frame)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (np.sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (np.sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    for window in [30, 60]:
        ws = _scale(window, fps)
        X[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        X[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()  # decreasing distance => positive approach
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    X[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = np.sqrt(A_vx**2 + A_vy**2)
        B_sp = np.sqrt(B_vx**2 + B_vy**2)
        X[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    return X

def transform_single(single_mouse, body_parts_tracked, fps):
    """Enhanced single mouse transform (FPS-aware windows/lags; distances in cm)."""
    available_body_parts = single_mouse.columns.get_level_values(0)

    # Base distance features (squared distances across body parts)
    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        lag = _scale(10, fps)
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag)
        speeds = pd.DataFrame({
            'sp_lf': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
            'sp_rt': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
            'sp_lf2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
            'sp_rt2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Body angle (orientation)
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

    # Core temporal features (windows scaled by fps)
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'cx_m{w}'] = cx.rolling(ws, **roll).mean()
            X[f'cy_m{w}'] = cy.rolling(ws, **roll).mean()
            X[f'cx_s{w}'] = cx.rolling(ws, **roll).std()
            X[f'cy_s{w}'] = cy.rolling(ws, **roll).std()
            X[f'x_rng{w}'] = cx.rolling(ws, **roll).max() - cx.rolling(ws, **roll).min()
            X[f'y_rng{w}'] = cy.rolling(ws, **roll).max() - cy.rolling(ws, **roll).min()
            X[f'disp{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).sum()**2 +
                                     cy.diff().rolling(ws, min_periods=1).sum()**2)
            X[f'act{w}'] = np.sqrt(cx.diff().rolling(ws, min_periods=1).var() +
                                   cy.diff().rolling(ws, min_periods=1).var())

        # Advanced features (fps-scaled)
        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

    # Nose-tail features with duration-aware lags
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    # Ear features with duration-aware offsets
    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-20, -10, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)  
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Enhanced pair transform (FPS-aware windows/lags; distances in cm)."""
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    # Inter-mouse distances (squared distances across all part pairs)
    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

    # Speed-like features via lagged displacements (duration-aware lag)
    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)
        shA = mouse_pair['A']['ear_left'].shift(lag)
        shB = mouse_pair['B']['ear_left'].shift(lag)
        speeds = pd.DataFrame({
            'sp_A': np.square(mouse_pair['A']['ear_left'] - shA).sum(axis=1, skipna=False),
            'sp_AB': np.square(mouse_pair['A']['ear_left'] - shB).sum(axis=1, skipna=False),
            'sp_B': np.square(mouse_pair['B']['ear_left'] - shB).sum(axis=1, skipna=False),
        })
        X = pd.concat([X, speeds], axis=1)

    if 'nose+tail_base' in X.columns and 'ear_left+ear_right' in X.columns:
        X['elong'] = X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6)

    # Relative orientation
    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    # Approach rate (duration-aware lag)
    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    # Distance bins (cm; unchanged by fps)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

    # Temporal interaction features (fps-adjusted windows)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd_full = np.square(mouse_pair['A']['body_center'] - mouse_pair['B']['body_center']).sum(axis=1, skipna=False)

        for w in [5, 15, 30, 60]:
            ws = _scale(w, fps)
            roll = dict(min_periods=1, center=True)
            X[f'd_m{w}']  = cd_full.rolling(ws, **roll).mean()
            X[f'd_s{w}']  = cd_full.rolling(ws, **roll).std()
            X[f'd_mn{w}'] = cd_full.rolling(ws, **roll).min()
            X[f'd_mx{w}'] = cd_full.rolling(ws, **roll).max()

            d_var = cd_full.rolling(ws, **roll).var()
            X[f'int{w}'] = 1 / (1 + d_var)

            Axd = mouse_pair['A']['body_center']['x'].diff()
            Ayd = mouse_pair['A']['body_center']['y'].diff()
            Bxd = mouse_pair['B']['body_center']['x'].diff()
            Byd = mouse_pair['B']['body_center']['y'].diff()
            coord = Axd * Bxd + Ayd * Byd
            X[f'co_m{w}'] = coord.rolling(ws, **roll).mean()
            X[f'co_s{w}'] = coord.rolling(ws, **roll).std()

    # Nose-nose dynamics (duration-aware lags)
    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    # Velocity alignment (duration-aware offsets)
    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-20, -10, 0, 10, 20]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        # Advanced interaction (fps-adjusted internals)
        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)
# ==================== CROSS-VALIDATION REPORTING ====================

def run_group_cv_report(X_np, label_df, meta_df, base_estimator, actions, n_splits=5):
    """Run GroupKFold CV and build per-lab and macro F1 reports."""
    if 'lab_id' not in meta_df.columns:
        return None, None

    groups_all = meta_df['lab_id'].to_numpy()
    unique_groups = np.unique(groups_all)
    if len(unique_groups) < 2:
        return None, None

    per_lab_rows = []
    macro_rows = []

    for action in actions:
        y_raw = label_df[action].to_numpy()
        mask = ~pd.isna(y_raw)
        y_action = y_raw[mask].astype(int)
        if len(np.unique(y_action)) < 2:
            continue

        X_action = X_np[mask]
        g_action = groups_all[mask]
        action_groups = np.unique(g_action)
        splits = min(n_splits, len(action_groups))
        if splits < 2:
            continue

        cv = GroupKFold(n_splits=splits)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(X_action, y_action, g_action)):
            estimator = clone(base_estimator)
            estimator.fit(X_action[train_idx], y_action[train_idx])
            preds = estimator.predict(X_action[valid_idx])

            y_valid = y_action[valid_idx]
            g_valid = g_action[valid_idx]

            fold_macro = f1_score(y_valid, preds, average='macro', zero_division=0)
            macro_rows.append({
                'action': action,
                'fold': fold_idx,
                'macro_f1': fold_macro,
            })

            for lab in np.unique(g_valid):
                lab_mask = g_valid == lab
                lab_f1 = f1_score(
                    y_valid[lab_mask], preds[lab_mask], average='macro', zero_division=0
                )
                per_lab_rows.append({
                    'action': action,
                    'lab_id': lab,
                    'fold': fold_idx,
                    'macro_f1': lab_f1,
                })

    if not per_lab_rows and not macro_rows:
        return None, None

    per_lab_df = pd.DataFrame(per_lab_rows) if per_lab_rows else None
    macro_df = pd.DataFrame(macro_rows) if macro_rows else None
    return per_lab_df, macro_df
# ==================== ENSEMBLE TRAINING ====================

def submit_ensemble(body_parts_tracked_str, switch_tr, X_tr, label, meta, n_samples):
    models = []

    models.append(make_pipeline(
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=225, learning_rate=0.07, min_child_samples=40,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8, verbose=-1,
                device='gpu', gpu_platform_id=0, gpu_device_id=0,
                random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED
            ), int(n_samples/1.3),
        )
    ))
    models.append(make_pipeline(
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=150, learning_rate=0.1, min_child_samples=20,
                num_leaves=63, max_depth=8, subsample=0.7, colsample_bytree=0.9,
                reg_alpha=0.1, reg_lambda=0.1, verbose=-1,
                device='gpu', gpu_platform_id=0, gpu_device_id=0,
                random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED
            ), int(n_samples/2),
        )
    ))
    models.append(make_pipeline(
        StratifiedSubsetClassifier(
            lightgbm.LGBMClassifier(
                n_estimators=100, learning_rate=0.05, min_child_samples=30,
                num_leaves=127, max_depth=10, subsample=0.75, verbose=-1,
                device='gpu', gpu_platform_id=0, gpu_device_id=0,
                random_state=SEED, bagging_seed=SEED, feature_fraction_seed=SEED, data_random_seed=SEED
            ), int(n_samples/3),
        )
    ))
    if XGBOOST_AVAILABLE:
        models.append(make_pipeline(
            StratifiedSubsetClassifier(
                XGBClassifier(
                    n_estimators=180, learning_rate=0.08, max_depth=6,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    tree_method='gpu_hist', predictor='gpu_predictor', verbosity=0,
                    random_state=SEED
                ), int(n_samples/1.5),
            )
        ))
    if CATBOOST_AVAILABLE:
        models.append(make_pipeline(
            StratifiedSubsetClassifier(
                CatBoostClassifier(
                    iterations=120, learning_rate=0.1, depth=6,
                    verbose=False, allow_writing_files=False,
                    task_type='GPU', devices='0',
                    random_seed=SEED
                ), n_samples,
            )
        ))

    X_tr_np = X_tr.to_numpy(np.float32, copy=False)

    if models:
        base_estimator = models[0]
        per_lab_df, macro_df = run_group_cv_report(
            X_tr_np, label, meta, base_estimator, label.columns
        )
        if per_lab_df is not None:
            per_lab_df['body_parts'] = body_parts_tracked_str
            per_lab_df['mode'] = switch_tr
            cv_per_lab_reports.append(per_lab_df)
        if macro_df is not None:
            macro_df['body_parts'] = body_parts_tracked_str
            macro_df['mode'] = switch_tr
            cv_macro_reports.append(macro_df)

    del X_tr; gc.collect()

    model_list = []
    for action in label.columns:
        y_raw = label[action].to_numpy()
        mask = ~pd.isna(y_raw)
        y_action = y_raw[mask].astype(int)
        if not (y_action == 0).all() and np.sum(y_action) >= 5:
            trained = []
            idx = np.flatnonzero(mask)
            for m in models:
                m_clone = clone(m)
                m_clone.fit(X_tr_np[idx], y_action)
                trained.append(m_clone)
            model_list.append((action, trained))

    del X_tr_np; gc.collect()

    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 'test',
        generate_single=(switch_tr == 'single'),
        generate_pair=(switch_tr == 'pair')
    )

    fps_lookup = (
        test_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    if verbose:
        print(f"n_videos: {len(test_subset)}, n_models: {len(models)}")

    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)

            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i).astype(np.float32)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i).astype(np.float32)

            X_te_np = X_te.to_numpy(np.float32, copy=False)
            del X_te, data_te; gc.collect()

            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, trained in model_list:
                if action in actions_te:
                    probs = [m.predict_proba(X_te_np)[:, 1] for m in trained]
                    pred[action] = np.mean(probs, axis=0)

            del X_te_np; gc.collect()

            if pred.shape[1] != 0:
                sub_part = predict_multiclass_adaptive(pred, meta_te, action_thresholds)
                submission_list.append(sub_part)
            else:
                if verbose:
                    print("  ERROR: no training data")

        except Exception as e:
            if verbose:
                print(f"  ERROR: {str(e)[:50]}")
            try:
                del data_te
            except Exception:
                pass
            gc.collect()

def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    submission = submission[submission.start_frame < submission.stop_frame]

    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop:
                mask[i] = False
            else:
                last_stop = row['stop_frame']
        group_list.append(group[mask])
    submission = pd.concat(group_list) if group_list else submission

    s_list = []
    for _, row in dataset.iterrows():
        lab_id = row['lab_id']
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue

        if verbose:
            print(f"Video {video_id} has no predictions")

        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)

        vid_behaviors = eval(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])

        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1

        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_len = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_len
                batch_stop = min(batch_start + batch_len, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])

    submission = submission.reset_index(drop=True)
    return submission
# ==================== MAIN LOOP ====================

submission_list = []

print(f"XGBoost: {XGBOOST_AVAILABLE}, CatBoost: {CATBOOST_AVAILABLE}\n")

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}. Processing: {len(body_parts_tracked)} body parts")
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]

        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

        _fps_lookup = (
            train_subset[['video_id', 'frames_per_second']]
            .drop_duplicates('video_id')
            .set_index('video_id')['frames_per_second']
            .to_dict()
        )

        single_list, single_label_list, single_meta_list = [], [], []
        pair_list, pair_label_list, pair_meta_list = [], [], []

        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_list.append(data)
                single_meta_list.append(meta)
                single_label_list.append(label)
            else:
                pair_list.append(data)
                pair_meta_list.append(meta)
                pair_label_list.append(label)

        if len(single_list) > 0:
            single_feats_parts = []
            for data_i, meta_i in zip(single_list, single_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(Xi)

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
 
            single_label = pd.concat(single_label_list, axis=0, ignore_index=True)
            single_meta  = pd.concat(single_meta_list,  axis=0, ignore_index=True)

            del single_list, single_label_list, single_meta_list, single_feats_parts
            gc.collect()

            print(f"  Single: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'single', X_tr, single_label, single_meta, 2_000_000)

            del X_tr, single_label, single_meta
            gc.collect()

        if len(pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(pair_list, pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                Xi = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(Xi)

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)

            
            pair_label = pd.concat(pair_label_list, axis=0, ignore_index=True)
            pair_meta  = pd.concat(pair_meta_list,  axis=0, ignore_index=True)

            del pair_list, pair_label_list, pair_meta_list, pair_feats_parts
            gc.collect()

            print(f"  Pair: {X_tr.shape}")
            submit_ensemble(body_parts_tracked_str, 'pair', X_tr, pair_label, pair_meta, 900_000)

            del X_tr, pair_label, pair_meta
            gc.collect()

    except Exception as e:
        print(f'***Exception*** {str(e)[:100]}')

    gc.collect()
    print()

if cv_per_lab_reports:
    per_lab_report_df = pd.concat(cv_per_lab_reports, ignore_index=True)
    per_lab_summary_df = (
        per_lab_report_df
        .groupby(['mode', 'body_parts', 'lab_id'])['macro_f1']
        .mean()
        .reset_index(name='mean_macro_f1')
    )
    per_lab_report_df.to_csv('cv_per_lab_report.csv', index=False)
    per_lab_summary_df.to_csv('cv_per_lab_summary.csv', index=False)
    print('Saved per-lab CV report to cv_per_lab_report.csv and summary to cv_per_lab_summary.csv')

if cv_macro_reports:
    macro_report_df = pd.concat(cv_macro_reports, ignore_index=True)
    macro_summary_df = (
        macro_report_df
        .groupby(['mode', 'body_parts', 'action'])['macro_f1']
        .mean()
        .reset_index(name='mean_macro_f1')
    )
    overall_macro_df = (
        macro_summary_df
        .groupby(['mode', 'body_parts'])['mean_macro_f1']
        .mean()
        .reset_index(name='overall_macro_f1')
    )
    macro_report_df.to_csv('cv_macro_f1_report.csv', index=False)
    macro_summary_df.to_csv('cv_macro_f1_summary.csv', index=False)
    overall_macro_df.to_csv('cv_macro_f1_overall.csv', index=False)
    print('Saved macro F1 CV report to cv_macro_f1_report.csv with summaries')

if len(submission_list) > 0:
    submission = pd.concat(submission_list, ignore_index=True)
else:
    submission = pd.DataFrame({
        'video_id': [438887472],
        'agent_id': ['mouse1'],
        'target_id': ['self'],
        'action': ['rear'],
        'start_frame': [278],
        'stop_frame': [500]
    })

submission_robust = robustify(submission, test, 'test')
submission_robust.index.name = 'row_id'
submission_robust.to_csv('submission.csv')
print(f"\nSubmission created: {len(submission_robust)} predictions")
