# MABE2025 Mouse Behavior Detection

## Recent Updates

### 5-Fold Cross-Validation with GPU Support

This update adds comprehensive cross-validation and GPU acceleration to the training pipeline:

#### Features:
- **5-Fold Cross-Validation (Actions)**: Evaluate model performance with K-fold CV to prevent overfitting
- **Leave-One-Lab-Out CV**: Test generalization across different labs
- **GPU Acceleration**: Full GPU support for LightGBM, XGBoost, and CatBoost
- **Per-Action F1 Scores**: Track F1 score for each behavior action during training
- **Per-Lab F1 Scores**: Evaluate performance on each lab separately
- **Macro F1 Scores**: Overall performance metrics averaged across actions and labs
- **Configurable**: Easy toggle between CPU/GPU and CV on/off

#### Configuration

Edit the following variables in `train_v1.py`:

```python
# Cross-validation configuration
USE_CROSS_VALIDATION = True  # Set to False to disable action-wise CV
N_CV_FOLDS = 5              # Number of cross-validation folds
USE_LAB_CV = True           # Set to False to disable lab-wise CV

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

#### Lab-wise Cross-Validation

After action-wise CV, you'll see leave-one-lab-out CV results:

```
============================================================
Leave-One-Lab-Out Cross-Validation
============================================================
Total labs: 5
Labs: lab1, lab2, lab3, lab4, lab5
Strategy: Train on 4 labs, validate on 1 lab
============================================================

Fold 1/5: Validating on lab 'lab1'
  Training on 4 labs: lab2, lab3, lab4, lab5
  Validation samples: 45231
  Lab 'lab1' F1: 0.7234 (avg across 3 actions)

Fold 2/5: Validating on lab 'lab2'
  Training on 4 labs: lab1, lab3, lab4, lab5
  Validation samples: 52341
  Lab 'lab2' F1: 0.7456 (avg across 3 actions)

...

============================================================
Lab-wise Cross-Validation Results
============================================================
Lab ID                         F1 Score
------------------------------------------
lab1                           0.7234
lab2                           0.7456
lab3                           0.7123
lab4                           0.7389
lab5                           0.7298
------------------------------------------
Macro F1 (across labs)         0.7300
============================================================
```

This shows how well your model generalizes to **unseen labs**, which is crucial for real-world deployment!

#### GPU Requirements

To use GPU acceleration, ensure you have:
- CUDA-compatible GPU
- LightGBM compiled with GPU support
- XGBoost with GPU support (optional)
- CatBoost with GPU support (optional)

**GPU Error Handling:**
- The code includes automatic CPU fallback if GPU training fails
- GPU uses higher `min_child_samples` (+20) to avoid split errors
- If a GPU error is detected, the model automatically retries with CPU
- Warnings about GPU failures will be displayed but won't stop training

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

The code now tracks **four types** of F1 scores for comprehensive evaluation:

1. **Per-action F1 (during action CV)**: F1 score for each behavior during 5-fold CV
   - Example: "rear: 0.7256 ± 0.0085"

2. **Macro F1 (across actions)**: Average F1 across all behaviors
   - Example: "Macro F1 (avg): 0.7683"

3. **Per-lab F1 (during lab CV)**: F1 score when validating on each lab
   - Example: "Lab 'lab1' F1: 0.7234"

4. **Macro F1 (across labs)**: Average F1 across all labs (generalization metric!)
   - Example: "Macro F1 (across labs): 0.7300"

### Why Lab CV Matters

Leave-one-lab-out cross-validation is crucial because:
- **Tests generalization** to completely unseen lab environments
- **Detects overfitting** to specific lab conditions or equipment
- **Validates robustness** across different experimental setups
- **Predicts real-world performance** when deploying to new labs

A model might achieve high action-wise CV scores but fail on new labs if it's overfitting to lab-specific artifacts!
