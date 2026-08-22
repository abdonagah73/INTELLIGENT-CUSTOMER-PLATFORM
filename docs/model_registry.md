# Model Registry Documentation

This document describes the versioning schema, promotion criteria, and rollback procedures for the `churn_predictor` model registry in MLflow and Azure ML Registry.

---

## 1. Model Naming & Versioning Schema

- **Model Name**: `churn_predictor` (Standardized across all systems)
- **Version Tagging**:
  - `v1.0`, `v1.1`, `v1.2`...
  - **Major Version change (e.g. v1.x to v2.x)**: Algorithm change (e.g., switching from XGBoost to Gradient Boosting or a Temporal Fusion Transformer).
  - **Minor Version change (e.g. v1.0 to v1.1)**: Model retrained on newer feature snapshot with identical architecture.

### Model Metadata Fields
Each registered version must record:
- **Feature Snapshot Date**: (e.g. `2024-03-01`)
- **Optimal Threshold**: Classification boundary computed from cost tuning.
- **Evaluation Metrics**: Test ROC-AUC, test PR-AUC, test Recall at optimal threshold.
- **Run ID**: Link to the MLflow execution run.

---

## 2. Model Promotion Criteria

To promote a model candidate from **Staging** to **Production**, the candidate must satisfy the following automated gating criteria:

1. **ROC-AUC Gate**: Test ROC-AUC must be greater than or equal to `0.84`.
2. **PR-AUC Gate**: Test PR-AUC must be within 2% or exceed the current production model's PR-AUC.
3. **Recall Gate**: Recall at the optimized threshold must exceed `0.80`.
4. **Performance Safety Check**: In automated retraining, the new model's ROC-AUC must not drop by more than 2% compared to the production model baseline (failsafe to prevent regression).
5. **API Contract Compatibility**: Must pass the automated REST API integration test suite (10 test requests with correct schema and shape).

---

## 3. Rollback Procedure

If a model degradation or serving error occurs in production, the rollback procedure is executed instantly:

### Direct MLflow Command Rollback
To demote a model version and restore the previous baseline, update the stages in the MLflow client:

```python
import mlflow
client = mlflow.tracking.MlflowClient()

# Demote current model to Archived
client.transition_model_version_stage(
    name="churn_predictor",
    version=2,  # Current buggy version
    stage="Archived"
)

# Promote previous model to Production
client.transition_model_version_stage(
    name="churn_predictor",
    version=1,  # Previous stable version
    stage="Production"
)
```

### CLI Rollback (via Azure ML CLI)
If deployed as a managed endpoint, traffic can be routed back using the Azure CLI:
```bash
az ml online-endpoint update --name customer-churn-endpoint --traffic "churn-predictor-v1=100, churn-predictor-v2=0"
```
This routes 100% of the live API traffic back to version 1 instantly with zero downtime.
