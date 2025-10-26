# MABE2025 Mouse Behavior Detection

## Recent Updates

### 5-Fold Cross-Validation with GPU Support

This update adds comprehensive cross-validation and GPU acceleration to the training pipeline:

#### Features:
- **5-Fold Cross-Validation**: Evaluate model performance with K-fold CV to prevent overfitting
- **GPU Acceleration**: Full GPU support for LightGBM, XGBoost, and CatBoost
- **Per-Action F1 Scores**: Track F1 score for each behavior action during training
- **Macro F1 Score**: Overall performance metric averaged across all actions
- **Lab-wise F1 Scores**: Calculate F1 scores separately for each lab
- **Configurable**: Easy toggle between CPU/GPU and CV on/off

#### Configuration

Edit the following variables in `train_v1.py`:

```python
# Cross-validation configuration
USE_CROSS_VALIDATION = True  # Set to False to disable CV
N_CV_FOLDS = 5              # Number of cross-validation folds

# GPU configuration
USE_GPU = True  # Set to False to use CPU
```

#### Cross-Validation Output

During training, you'll see detailed CV results:

```
============================================================
Starting 5-Fold Cross-Validation
============================================================

Action: rear (positives: 1523/10245)
  Fold 1: F1 = 0.7234
  Fold 2: F1 = 0.7156
  Fold 3: F1 = 0.7389
  Fold 4: F1 = 0.7201
  Fold 5: F1 = 0.7298
  Mean CV F1: 0.7256 ± 0.0085

Action: walk (positives: 2341/10245)
  Fold 1: F1 = 0.8123
  Fold 2: F1 = 0.8098
  ...

============================================================
Cross-Validation Summary
============================================================
Action               Mean F1      Std F1
--------------------------------------------
rear                 0.7256       0.0085
walk                 0.8109       0.0034
...
--------------------------------------------
Macro F1 (avg)       0.7683
============================================================
```

#### Lab-wise Scores

After training, you'll see F1 scores broken down by lab:

```
============================================================
Lab-wise F1 Scores
============================================================
Lab ID                         F1 Score
------------------------------------------
lab1                           0.7234
lab2                           0.7456
lab3                           0.7123
------------------------------------------
Macro F1 (across labs)         0.7271
============================================================
```

#### GPU Requirements

To use GPU acceleration, ensure you have:
- CUDA-compatible GPU
- LightGBM compiled with GPU support
- XGBoost with GPU support (optional)
- CatBoost with GPU support (optional)

If GPU is not available, set `USE_GPU = False` and the code will automatically use CPU.

#### Test Coverage

Test coverage: **60.59%** (412/680 statements)

Run tests with:
```bash
pytest tests/ --cov=train_v1 --cov-report=html
```

## Model Architecture

- **3 LightGBM models** with different hyperparameters
- **1 XGBoost model** (if available)
- **1 CatBoost model** (if available)
- Ensemble averaging across all models

## Performance Metrics

The code tracks three types of F1 scores:
1. **Per-action F1**: F1 score for each behavior (rear, walk, etc.)
2. **Macro F1 (actions)**: Average F1 across all actions
3. **Macro F1 (labs)**: Average F1 across all labs
