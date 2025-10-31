from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.base import clone
from xgboost import XGBClassifier
from tqdm.notebook import tqdm
import numpy as np
import itertools
import warnings
import optuna
import joblib
import glob
import gc
import os
import pandas as pd
import json

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

class CVTrainer:
    """Custom cross-validation trainer to replace koolbox.Trainer"""
    def __init__(self, estimator, cv, cv_args, metric, task="binary", verbose=False, save=True, save_path=None):
        self.estimator = estimator
        self.cv = cv
        self.cv_args = cv_args
        self.metric = metric
        self.task = task
        self.verbose = verbose
        self.save = save
        self.save_path = save_path
        self.oof_preds = None
        self.models = []

    def fit(self, X, y):
        """Perform cross-validation training"""
        # Initialize OOF predictions
        self.oof_preds = np.zeros(len(y), dtype=np.float32)
        self.models = []

        # Create save directory if needed
        if self.save and self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

        # Perform cross-validation
        groups = self.cv_args.get("groups", None)

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, y, groups=groups)):
            if self.verbose:
                print(f"  Training fold {fold}...")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Clone and train model
            model = clone(self.estimator)
            model.fit(X_train, y_train)

            # Generate OOF predictions
            if self.task == "binary":
                # For binary classification, use predict_proba
                oof_pred = model.predict_proba(X_val)[:, 1]
            else:
                oof_pred = model.predict(X_val)

            self.oof_preds[val_idx] = oof_pred
            self.models.append(model)

            # Save fold model
            if self.save and self.save_path:
                fold_path = os.path.join(self.save_path, f"fold_{fold}.pkl")
                joblib.dump(model, fold_path)

        # Save the trainer object itself with a pattern matching *_trainer_*.pkl
        if self.save and self.save_path:
            trainer_path = os.path.join(self.save_path, "cv_trainer_ensemble.pkl")
            joblib.dump(self, trainer_path)

        return self

    def predict(self, X):
        """Generate predictions by averaging across all fold models"""
        if not self.models:
            # If models not in memory, load from disk
            if self.save_path and os.path.exists(self.save_path):
                fold_files = glob.glob(os.path.join(self.save_path, "fold_*.pkl"))
                self.models = [joblib.load(f) for f in sorted(fold_files)]

        if not self.models:
            raise ValueError("No trained models available for prediction")

        # Average predictions across folds
        predictions = []
        for model in self.models:
            if self.task == "binary":
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        return np.mean(predictions, axis=0)

class CFG:
    train_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
    test_path = "/kaggle/input/MABe-mouse-behavior-detection/test.csv"
    train_annotation_path = "/kaggle/input/MABe-mouse-behavior-detection/train_annotation"
    train_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/train_tracking"
    test_tracking_path = "/kaggle/input/MABe-mouse-behavior-detection/test_tracking"

    model_path = "/kaggle/input/social-action-recognition-in-mice-xgboost"
    model_name = "xgboost"
    
    # mode = "validate"
    mode = "submit"
    
    n_splits = 3
    cv = StratifiedGroupKFold(n_splits)

    model = XGBClassifier(
        verbosity=0, 
        random_state=42,
        n_estimators=250, 
        learning_rate=0.08, 
        max_depth=6,
        min_child_weight=5, 
        subsample=0.8, 
        colsample_bytree=0.8
    )
train = pd.read_csv(CFG.train_path)
train['n_mice'] = 4 - train[['mouse1_strain', 'mouse2_strain', 'mouse3_strain', 'mouse4_strain']].isna().sum(axis=1)
train_without_mabe22 = train.query("~lab_id.str.startswith('MABe22_')")

test = pd.read_csv(CFG.test_path)
body_parts_tracked_list = list(np.unique(train.body_parts_tracked))
# Creating solution data
def create_solution_df(dataset):
    solution = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
    
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'): 
            continue
        
        video_id = row['video_id']
        path = f"{CFG.train_annotation_path}/{lab_id}/{video_id}.parquet"
        try:
            annot = pd.read_parquet(path)
        except FileNotFoundError:
            continue
    
        annot['lab_id'] = lab_id
        annot['video_id'] = video_id
        annot['behaviors_labeled'] = row['behaviors_labeled']
        annot['target_id'] = np.where(annot.target_id != annot.agent_id, annot['target_id'].apply(lambda s: f"mouse{s}"), 'self')
        annot['agent_id'] = annot['agent_id'].apply(lambda s: f"mouse{s}")
        solution.append(annot)
    
    solution = pd.concat(solution)
    
    return solution

if CFG.mode == 'validate':
    solution = create_solution_df(train_without_mabe22)
drop_body_parts =  [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2', 'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

def generate_mouse_data(dataset, traintest, traintest_directory=None, generate_single=True, generate_pair=True):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"
        
    for _, row in dataset.iterrows():
        lab_id = row.lab_id
        if lab_id.startswith('MABe22') or type(row.behaviors_labeled) != str: 
            continue
        
        video_id = row.video_id
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
        if len(np.unique(vid.bodypart)) > 5:
            vid = vid.query("~ bodypart.isin(@drop_body_parts)")
        pvid = vid.pivot(columns=['mouse_id', 'bodypart'], index='video_frame', values=['x', 'y'])
        
        del vid
        gc.collect()
        
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
                        'video_frame': single_mouse.index
                    })
                    if traintest == 'train':
                        single_mouse_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=single_mouse.index)
                        annot_subset = annot.query("(agent_id == @mouse_id) & (target_id == @mouse_id)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            single_mouse_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'single', single_mouse, single_mouse_meta, single_mouse_label
                    else:
                        yield 'single', single_mouse, single_mouse_meta, vid_agent_actions
                except KeyError:
                    pass

        if generate_pair:
            vid_behaviors_subset = vid_behaviors.query("target != 'self'")
            if len(vid_behaviors_subset) > 0:
                for agent, target in itertools.permutations(np.unique(pvid.columns.get_level_values('mouse_id')), 2): # int8
                    agent_str = f"mouse{agent}"
                    target_str = f"mouse{target}"
                    vid_agent_actions = np.unique(vid_behaviors_subset.query("(agent == @agent_str) & (target == @target_str)").action)
                    mouse_pair = pd.concat([pvid[agent], pvid[target]], axis=1, keys=['A', 'B'])
                    assert len(mouse_pair) == len(pvid)
                    mouse_pair_meta = pd.DataFrame({
                        'video_id': video_id,
                        'agent_id': agent_str,
                        'target_id': target_str,
                        'video_frame': mouse_pair.index
                    })
                    if traintest == 'train':
                        mouse_pair_label = pd.DataFrame(0.0, columns=vid_agent_actions, index=mouse_pair.index)
                        annot_subset = annot.query("(agent_id == @agent) & (target_id == @target)")
                        for i in range(len(annot_subset)):
                            annot_row = annot_subset.iloc[i]
                            mouse_pair_label.loc[annot_row['start_frame']:annot_row['stop_frame'], annot_row.action] = 1.0
                        yield 'pair', mouse_pair, mouse_pair_meta, mouse_pair_label
                    else:
                        yield 'pair', mouse_pair, mouse_pair_meta, vid_agent_actions
def safe_rolling(series, window, func, min_periods=None):
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
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
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = np.sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)

    for w in [25, 50, 75]:
        ws = _scale(w, fps)
        X[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 5)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    X[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 5)).sum()

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)

    scales = [20, 40, 60, 80]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            X[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            X[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).std()

    if len(scales) >= 2 and f'sp_m{scales[0]}' in X.columns and f'sp_m{scales[-1]}' in X.columns:
        X['sp_ratio'] = X[f'sp_m{scales[0]}'] / (X[f'sp_m{scales[-1]}'] + 1e-6)

    return X

def add_state_features(X, center_x, center_y, fps):
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        for window in [20, 40, 60, 80]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    X[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 5)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                X[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 5)).sum()
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            X[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            X[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    for span in [30, 60, 120]:
        s = _scale(span, fps)
        X[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        X[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [30, 60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            X[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    return X

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = np.sqrt(rel_x**2 + rel_y**2)

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

    approach = -rel_dist.diff()
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
    available_body_parts = single_mouse.columns.get_level_values(0)

    X = pd.DataFrame({
        f"{p1}+{p2}": np.square(single_mouse[p1] - single_mouse[p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.combinations(body_parts_tracked, 2)
        if p1 in available_body_parts and p2 in available_body_parts
    })
    X = X.reindex(columns=[f"{p1}+{p2}" for p1, p2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

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

    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose'] - single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        X['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (
            np.sqrt(v1['x']**2 + v1['y']**2) * np.sqrt(v2['x']**2 + v2['y']**2) + 1e-6)

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

        X = add_curvature_features(X, cx, cy, fps)
        X = add_multiscale_features(X, cx, cy, fps)
        X = add_state_features(X, cx, cy, fps)
        X = add_longrange_features(X, cx, cy, fps)

    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = np.sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nt_lg{lag}'] = nt_dist.shift(l)
            X[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_d = np.sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 +
                        (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        for off in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'ear_o{off}'] = ear_d.shift(-o)
        w = _scale(30, fps)
        X['ear_con'] = ear_d.rolling(w, min_periods=1, center=True).std() / \
                       (ear_d.rolling(w, min_periods=1, center=True).mean() + 1e-6)

    return X.astype(np.float32, copy=False)

def transform_pair(mouse_pair, body_parts_tracked, fps):
    avail_A = mouse_pair['A'].columns.get_level_values(0)
    avail_B = mouse_pair['B'].columns.get_level_values(0)

    X = pd.DataFrame({
        f"12+{p1}+{p2}": np.square(mouse_pair['A'][p1] - mouse_pair['B'][p2]).sum(axis=1, skipna=False)
        for p1, p2 in itertools.product(body_parts_tracked, repeat=2)
        if p1 in avail_A and p2 in avail_B
    })
    X = X.reindex(columns=[f"12+{p1}+{p2}" for p1, p2 in itertools.product(body_parts_tracked, repeat=2)], copy=False)

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

    if all(p in avail_A for p in ['nose', 'tail_base']) and all(p in avail_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        X['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            np.sqrt(dir_A['x']**2 + dir_A['y']**2) * np.sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    if all(p in avail_A for p in ['nose']) and all(p in avail_B for p in ['nose']):
        cur = np.square(mouse_pair['A']['nose'] - mouse_pair['B']['nose']).sum(axis=1, skipna=False)
        lag = _scale(10, fps)
        shA_n = mouse_pair['A']['nose'].shift(lag)
        shB_n = mouse_pair['B']['nose'].shift(lag)
        past = np.square(shA_n - shB_n).sum(axis=1, skipna=False)
        X['appr'] = cur - past

    if 'body_center' in avail_A and 'body_center' in avail_B:
        cd = np.sqrt((mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x'])**2 +
                     (mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y'])**2)
        X['v_cls'] = (cd < 5.0).astype(float)
        X['cls']   = ((cd >= 5.0) & (cd < 15.0)).astype(float)
        X['med']   = ((cd >= 15.0) & (cd < 30.0)).astype(float)
        X['far']   = (cd >= 30.0).astype(float)

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

    if 'nose' in avail_A and 'nose' in avail_B:
        nn = np.sqrt((mouse_pair['A']['nose']['x'] - mouse_pair['B']['nose']['x'])**2 +
                     (mouse_pair['A']['nose']['y'] - mouse_pair['B']['nose']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            X[f'nn_lg{lag}']  = nn.shift(l)
            X[f'nn_ch{lag}']  = nn - nn.shift(l)
            is_cl = (nn < 10.0).astype(float)
            X[f'cl_ps{lag}']  = is_cl.rolling(l, min_periods=1).mean()

    if 'body_center' in avail_A and 'body_center' in avail_B:
        Avx = mouse_pair['A']['body_center']['x'].diff()
        Avy = mouse_pair['A']['body_center']['y'].diff()
        Bvx = mouse_pair['B']['body_center']['x'].diff()
        Bvy = mouse_pair['B']['body_center']['y'].diff()
        val = (Avx * Bvx + Avy * Bvy) / (np.sqrt(Avx**2 + Avy**2) * np.sqrt(Bvx**2 + Bvy**2) + 1e-6)

        for off in [-30, -20, -10, 0, 10, 20, 30]:
            o = _scale_signed(off, fps)
            X[f'va_{off}'] = val.shift(-o)

        w = _scale(30, fps)
        X['int_con'] = cd_full.rolling(w, min_periods=1, center=True).std() / \
                       (cd_full.rolling(w, min_periods=1, center=True).mean() + 1e-6)

        X = add_interaction_features(X, mouse_pair, avail_A, avail_B, fps)

    return X.astype(np.float32, copy=False)
def robustify(submission, dataset, traintest, traintest_directory=None):
    if traintest_directory is None:
        traintest_directory = f"/kaggle/input/MABe-mouse-behavior-detection/{traintest}_tracking"

    old_submission = submission.copy()
    submission = submission[submission.start_frame < submission.stop_frame]
    if len(submission) != len(old_submission):
        print("ERROR: Dropped frames with start >= stop")
    
    old_submission = submission.copy()
    group_list = []
    for _, group in submission.groupby(['video_id', 'agent_id', 'target_id']):
        group = group.sort_values('start_frame')
        mask = np.ones(len(group), dtype=bool)
        last_stop_frame = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row['start_frame'] < last_stop_frame:
                mask[i] = False
            else:
                last_stop_frame = row['stop_frame']
        group_list.append(group[mask])
        
    submission = pd.concat(group_list)
    
    if len(submission) != len(old_submission):
        print("ERROR: Dropped duplicate frames")
        
    s_list = []
    for idx, row in dataset.iterrows():
        lab_id = row['lab_id']
        if lab_id.startswith('MABe22'):
            continue
        
        video_id = row['video_id']
        if (submission.video_id == video_id).any():
            continue
        
        if type(row.behaviors_labeled) != str:
            continue

        print(f"Video {video_id} has no predictions.")
        
        path = f"{traintest_directory}/{lab_id}/{video_id}.parquet"
        vid = pd.read_parquet(path)
    
        vid_behaviors = json.loads(row['behaviors_labeled'])
        vid_behaviors = sorted(list({b.replace("'", "") for b in vid_behaviors}))
        vid_behaviors = [b.split(',') for b in vid_behaviors]
        vid_behaviors = pd.DataFrame(vid_behaviors, columns=['agent', 'target', 'action'])
    
        start_frame = vid.video_frame.min()
        stop_frame = vid.video_frame.max() + 1
    
        for (agent, target), actions in vid_behaviors.groupby(['agent', 'target']):
            batch_length = int(np.ceil((stop_frame - start_frame) / len(actions)))
            for i, (_, action_row) in enumerate(actions.iterrows()):
                batch_start = start_frame + i * batch_length
                batch_stop = min(batch_start + batch_length, stop_frame)
                s_list.append((video_id, agent, target, action_row['action'], batch_start, batch_stop))

    if len(s_list) > 0:
        submission = pd.concat([
            submission,
            pd.DataFrame(s_list, columns=['video_id', 'agent_id', 'target_id', 'action', 'start_frame', 'stop_frame'])
        ])
        print("ERROR: Filled empty videos")

    submission = submission.reset_index(drop=True)
    
    return submission
def predict_multiclass(pred, meta, thresholds):
    ama = np.argmax(pred.values, axis=1)
    max_proba = pred.max(axis=1).values

    threshold_array = np.array([thresholds.get(col, 0.27) for col in pred.columns])
    action_thresholds = threshold_array[ama]

    ama = np.where(max_proba >= action_thresholds, ama, -1)
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
        if stop_video_id[i] != video_id or stop_agent_id[i] != agent_id or stop_target_id[i] != target_id:
            new_stop_frame = meta.query("(video_id == @video_id)").video_frame.max() + 1
            submission_part.iat[i, submission_part.columns.get_loc('stop_frame')] = new_stop_frame

    return submission_part
def tune_threshold(oof_action, y_action):
    def objective(trial):
        threshold = trial.suggest_float("threshold", 0, 1, step=0.01)
        return f1_score(y_action, (oof_action >= threshold), zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000, n_jobs=-1)
    return study.best_params["threshold"]
def cross_validate_classifier(X, label, meta, body_parts_tracked_str, section):
    oof = pd.DataFrame(index=meta.video_frame)
    
    f1_list = []
    submission_list = []
    thresholds = {}
    
    for action in label.columns:
        action_mask = ~ label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        X_action = X[action_mask]
        groups_action = meta.video_id[action_mask]
        
        if len(np.unique(groups_action)) < CFG.n_splits:
            continue

        if not (y_action == 0).all():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    
                    trainer = CVTrainer(
                        estimator=clone(CFG.model),
                        cv=CFG.cv,
                        cv_args={"groups": groups_action},
                        metric=f1_score,
                        task="binary",
                        verbose=False,
                        save=True,
                        save_path=f"{CFG.model_name}/{section}/{action}"
                    )

                    trainer.fit(X_action, y_action)
                    oof_action = trainer.oof_preds

                    threshold = tune_threshold(oof_action, y_action)
                    thresholds[action] = threshold
            
                    f1 = f1_score(y_action, (oof_action >= threshold), zero_division=0)
                    f1_list.append((body_parts_tracked_str, action, f1))
                    
                    joblib.dump(oof_action, f"{CFG.model_name}/{section}/{action}/oof_pred_probs.pkl")
                    joblib.dump(threshold, f"{CFG.model_name}/{section}/{action}/threshold.pkl")
                    
                    print(f"\tF1: {f1:.4f} ({threshold:.2f}) Section: {section} Action: {action}")
    
                    del trainer
                    gc.collect()

            except Exception as e:
                oof_action = np.zeros(len(y_action))
                print(f"\tF1: 0.0000 (0.00) Section: {section} Action: {action}")
        
        else:
            oof_action = np.zeros(len(y_action))
            print(f"\tF1: 0.0000 (0.00) Section: {section} Action: {action}")
        
        oof_column = np.zeros(len(label))
        oof_column[action_mask] = oof_action
        oof[action] = oof_column

        del oof_action, action_mask, X_action, y_action, groups_action
        gc.collect()

    submission_part = predict_multiclass(oof, meta, thresholds)
    submission_list.append(submission_part)
    
    return submission_list, f1_list, thresholds
def submit(body_parts_tracked_str, switch_tr, section, thresholds):    
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
        
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(
        test_subset, 
        'test',
        generate_single=(switch_tr == 'single'), 
        generate_pair=(switch_tr == 'pair')
    )

    fps_lookup = (
        test_subset[['video_id', 'frames_per_second']]
        .drop_duplicates('video_id')
        .set_index('video_id')['frames_per_second']
        .to_dict()
    )

    submission_list = []
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup, default_fps=30.0)
            
            if switch_te == 'single':
                X_te = transform_single(data_te, body_parts_tracked, fps_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)
            del data_te
            gc.collect()
    
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action in actions_te:
                files = glob.glob(f"{CFG.model_path}/{CFG.model_name}/{section}/{action}/*_trainer_*.pkl")
                if len(files) == 1:
                    trainer = joblib.load(files[0])
                    pred[action] = trainer.predict(X_te)
                
                    del trainer
                    gc.collect()
                
            del X_te
            gc.collect()

            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te, thresholds)
                submission_list.append(submission_part)
                
        except KeyError:
            del data_te
            gc.collect()
            
    return submission_list
if CFG.mode == "validate":
    thresholds = {
        "single": {},
        "pair": {}
    }
else:
    thresholds = joblib.load(f"{CFG.model_path}/{CFG.model_name}/thresholds.pkl")
f1_list = []
submission_list = []

for section in range(1, len(body_parts_tracked_list)):
    body_parts_tracked_str = body_parts_tracked_list[section]
    try:
        body_parts_tracked = json.loads(body_parts_tracked_str)
        print(f"{section}/{len(body_parts_tracked_list)-1} Processing videos with: {body_parts_tracked}\n")
        
        if len(body_parts_tracked) > 5:
            body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
        train_subset = train[train.body_parts_tracked == body_parts_tracked_str]

        _fps_lookup = (
            train_subset[['video_id', 'frames_per_second']]
            .drop_duplicates('video_id')
            .set_index('video_id')['frames_per_second']
            .to_dict()
        )
        
        single_mouse_list = []
        single_mouse_label_list = []
        single_mouse_meta_list = []
        
        mouse_pair_list = []
        mouse_pair_label_list = []
        mouse_pair_meta_list = []
    
        for switch, data, meta, label in generate_mouse_data(train_subset, 'train'):
            if switch == 'single':
                single_mouse_list.append(data)
                single_mouse_meta_list.append(meta)
                single_mouse_label_list.append(label)
            else:
                mouse_pair_list.append(data)
                mouse_pair_meta_list.append(meta)
                mouse_pair_label_list.append(label)
            
            del data, meta, label
        gc.collect()
    
    
        if len(single_mouse_list) > 0:
            single_feats_parts = []
            for data_i, meta_i in zip(single_mouse_list, single_mouse_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                X_i = transform_single(data_i, body_parts_tracked, fps_i).astype(np.float32)
                single_feats_parts.append(X_i)
                del X_i, fps_i
            gc.collect()

            X_tr = pd.concat(single_feats_parts, axis=0, ignore_index=True)
            single_mouse_label = pd.concat(single_mouse_label_list, axis=0, ignore_index=True)
            single_mouse_meta = pd.concat(single_mouse_meta_list, axis=0, ignore_index=True)
            
            del single_feats_parts, single_mouse_list, single_mouse_label_list, single_mouse_meta_list
            gc.collect()

            if CFG.mode == 'validate':
                temp_submission_list, temp_f1_list, temp_thresholds = cross_validate_classifier(X_tr, single_mouse_label, single_mouse_meta, body_parts_tracked_str, section)
                
                if f"{section}" not in thresholds["single"].keys():
                    thresholds["single"][f"{section}"] = {}
                for k, v in temp_thresholds.items():
                    thresholds["single"][f"{section}"][k] = v                  
                
                f1_list.extend(temp_f1_list)
                submission_list.extend(temp_submission_list)
                
                del temp_submission_list, temp_f1_list, temp_thresholds, X_tr
                gc.collect()
            else:
                temp_submission_list = submit(body_parts_tracked_str, 'single', section, thresholds["single"][f"{section}"])
                submission_list.extend(temp_submission_list)
                
                del temp_submission_list, X_tr
                gc.collect()
                
        if len(mouse_pair_list) > 0:
            pair_feats_parts = []
            for data_i, meta_i in zip(mouse_pair_list, mouse_pair_meta_list):
                fps_i = _fps_from_meta(meta_i, _fps_lookup, default_fps=30.0)
                X_i = transform_pair(data_i, body_parts_tracked, fps_i).astype(np.float32)
                pair_feats_parts.append(X_i)
                del X_i, fps_i
            gc.collect()

            X_tr = pd.concat(pair_feats_parts, axis=0, ignore_index=True)
            mouse_pair_label = pd.concat(mouse_pair_label_list, axis=0, ignore_index=True)
            mouse_pair_meta = pd.concat(mouse_pair_meta_list, axis=0, ignore_index=True)
            
            del pair_feats_parts, mouse_pair_list, mouse_pair_label_list, mouse_pair_meta_list
            gc.collect()

            if CFG.mode == 'validate':
                temp_submission_list, temp_f1_list, temp_thresholds = cross_validate_classifier(X_tr, mouse_pair_label, mouse_pair_meta, body_parts_tracked_str, section)

                if f"{section}" not in thresholds["pair"].keys():
                    thresholds["pair"][f"{section}"] = {}
                for k, v in temp_thresholds.items():
                    thresholds["pair"][f"{section}"][k] = v  
                    
                f1_list.extend(temp_f1_list)
                submission_list.extend(temp_submission_list)
                
                del temp_submission_list, temp_f1_list, temp_thresholds, X_tr
                gc.collect()
            else:
                temp_submission_list = submit(body_parts_tracked_str, 'pair', section, thresholds["pair"][f"{section}"])
                
                submission_list.extend(temp_submission_list)
                del temp_submission_list, X_tr
                gc.collect()
                
    except Exception as e:
        print(f"\t{e}")
    print()



if CFG.mode == 'validate':  
    submission = pd.concat(submission_list)
    submission_robust = robustify(submission, train, 'train')
    print(f"Competition metric: {score(solution, submission_robust, ''):.4f}")

    f1_df = pd.DataFrame(f1_list, columns=['body_parts_tracked_str', 'action', 'binary F1 score'])
    print(f"Mean F1:            {f1_df['binary F1 score'].mean():.4f}")
  
    joblib.dump(thresholds, f"{CFG.model_name}/thresholds.pkl")
    joblib.dump(f1_df, f"{CFG.model_name}/scores.pkl")
if CFG.mode == 'submit':
    if len(submission_list) > 0:
        submission = pd.concat(submission_list)
    else:
        submission = pd.DataFrame(
            dict(
                video_id=438887472,
                agent_id='mouse1',
                target_id='self',
                action='rear',
                start_frame='278',
                stop_frame='500'
            ), index=[44])
        
    submission_robust = robustify(submission, test, 'test')
    submission_robust.index.name = 'row_id'
    submission_robust.to_csv('submission.csv')
    submission.head()
