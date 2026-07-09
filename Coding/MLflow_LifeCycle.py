import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pyfunc
from sklearn.metrics import silhouette_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger("MLflow")

# -------------------------------------------------------------------------
# 1. Custom MLflow PythonModel Wrapper for Production Deployment
# -------------------------------------------------------------------------
class RFMClusterWrapper(mlflow.pyfunc.PythonModel):
    """
    Production-grade MLflow wrapper encapsulating the entire inference pipeline.
    Guarantees that unseen incoming real-time requests (via FastAPI/Streamlit)
    undergo the exact mathematical state transformations without any data leakage.
    """
    def __init__(self, feature_columns, whale_threshold):
        self.feature_columns = feature_columns
        self.whale_threshold = whale_threshold

    def load_context(self, context):
        """
        Loads the saved processing blueprints and core estimator when the model spins up.
        """
        import joblib
        self.model = joblib.load(context.artifacts["model_path"])
        self.scaler = joblib.load(context.artifacts["scaler_path"])
        self.pca = joblib.load(context.artifacts["pca_path"])  # Fixed: Added PCA to the deployment context

    def predict(self, context, model_input):
        """
        Executes operational real-time scoring on incoming customer vectors.
        """
        import hdbscan
        import numpy as np
        import pandas as pd

        # Force input into standard pandas dataframe structural layout
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input, columns=self.feature_columns)

        # Step A: Apply non-parametric Log transformation to handle skewed inputs
        X = model_input[self.feature_columns].copy()
        X_log = np.log1p(X)
        
        # Step B: Project using the isolated parameters calculated during Training
        X_scaled = self.scaler.transform(X_log)
        
        # Step C: Reduce dimensions using the training eigenspace to fix dimensionality mismatch
        X_pca = self.pca.transform(X_scaled)  # Fixed: Ensured HDBSCAN receives correct 2D inputs

        # Ensure HDBSCAN approximate prediction caching mechanisms are fully active
        if not hasattr(self.model, 'prediction_data_') or self.model.prediction_data_ is None:
            self.model.generate_prediction_data()

        # Step D: Execute density approximate prediction mapping out out-of-sample points
        labels, probs = hdbscan.approximate_predict(self.model, X_pca)

        final_labels = labels.copy()
        final_probs = probs.copy()

        # Step E: Handle Out-of-Distribution points (Noise) using Soft Clustering Membership
        noise_mask = (labels == -1)
        if noise_mask.any():
            soft_scores = hdbscan.membership_vector(self.model, X_pca[noise_mask])
            final_labels[noise_mask] = np.argmax(soft_scores, axis=1)
            final_probs[noise_mask] = np.max(soft_scores, axis=1)

        # Step F: Construct business metrics profile outputs
        results = pd.DataFrame({
            "cluster_label": final_labels.astype(int),
            "cluster_probability": final_probs.astype(float),
            "is_noise": noise_mask,
            "is_whale": (noise_mask) & (model_input["Monetary"] >= self.whale_threshold)
        })

        logger.info(f"Segmented {len(results)} customers successfully.")
        return results.replace([np.inf, -np.inf, np.nan], 0)


# -------------------------------------------------------------------------
# 2. Main MLflow Tracking & Governance Orchestration Lifecycle
# -------------------------------------------------------------------------
def run_mlflow_lifecycle(rfm, rfm_scaled, rfm_melted, scaler, pca, final_model, min_c, min_s, best_m, m_input):
    """
    Orchestrates experiment registration, performance telemetry tracking, 
    artifact serialization, and automated model registry promotion gates.
    """
    logger.info("========== MLflow LifeCycle =========")

    experiment_name = "RFM_Customer_Segmentation"
    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    # Initiate MLflow tracking transaction run block
    with mlflow.start_run(run_name="HDBSCAN_Full_Lifecycle") as run:
        run_id = run.info.run_id

        # Log Hyperparameters mapping configurations
        params = {
            "input_metric": m_input,
            "selected_best_metric": best_m,
            "min_cluster_size": min_c,
            "min_samples": min_s,
            "cluster_selection_method": "eom"
        }
        mlflow.log_params(params)
        logger.info(params)

        labels = rfm['Cluster']
        mask = labels != -1

        # Calculate Unsupervised density separation evaluation metrics
        if len(set(labels)) > 1 and np.any(mask):
            try:
                dbcv_score = final_model.relative_validity_
                logger.info("DBCV Score calculated successfully.")
            except AttributeError as e:
                logger.error(f"DBCV calculation failed: {e}. Ensure gen_min_span_tree=True.")
                dbcv_score = 0

            s_score = silhouette_score(rfm_scaled[mask], labels[mask])
            noise_pct = (labels == -1).sum() / len(labels) * 100

            try:
                total_stability = final_model.cluster_persistence_.sum()
            except:
                total_stability = 0

            logger.info(f"Cluster Quality Metrics: DBCV={dbcv_score:.4f}, Silhouette={s_score:.4f}")
            logger.info(f"Noise Level: {noise_pct:.2f}% of total customers.")

            if noise_pct > 15:
                logger.warning("High noise level detected! Tuning recommended.")

            # Log operational metric tracking records
            mlflow.log_metrics({
                "DBCV": dbcv_score,
                "Silhouette": s_score,
                "Noise_Percentage": noise_pct,
                "Stability": total_stability
            })
        else:
            dbcv_score, s_score, noise_pct = 0, 0, 100

        # Create and track telemetry visualizations
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=rfm_melted, palette='viridis')
        plt.title(f"RFM Snake Plot (Best Metric: {best_m})")
        plt.tight_layout()
        plt.savefig("rfm_snake_plot.png")
        mlflow.log_artifact("rfm_snake_plot.png")
        plt.close()

        # Serialize system state artifacts to local binaries
        joblib.dump(final_model, "hdbscan_model.pkl")
        joblib.dump(scaler, "scaler.pkl")
        joblib.dump(pca, "pca.pkl")  # Fixed: Added serialization execution for PCA state matrix

        # Bundle structural pathways for deployment
        artifacts = {
            "model_path": "hdbscan_model.pkl", 
            "scaler_path": "scaler.pkl",
            "pca_path": "pca.pkl"    # Fixed: Injected PCA asset paths into artifact dictionary
        }

        input_example = rfm[['Recency', 'Frequency', 'Monetary']].iloc[:3]
        whale_threshold = rfm["Monetary"].quantile(0.95)

        # Define data typing expectations signature profiles
        signature = infer_signature(input_example, pd.DataFrame({
            "cluster_label": [0, 1, -1],
            "cluster_probability": [0.91, 0.87, 0.05],
            "is_noise": [False, False, True],
            "is_whale": [False, False, True]
        }))

        # Package full pipeline state under standard MLflow PyFunc structure
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=RFMClusterWrapper(
                feature_columns=['Recency', 'Frequency', 'Monetary'],
                whale_threshold=whale_threshold
            ),
            artifacts=artifacts,
            signature=signature,
            input_example=input_example
        )

    # -------------------------------------------------------------------------
    # 3. Model Governance: Automated Stage Transition Gates
    # -------------------------------------------------------------------------
    model_name = "RFM_Segmentation_Production"

    registered_model = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )

    logger.info(f"Model registered: name={registered_model.name}, version={registered_model.version}")
    latest_version = registered_model.version

    # Transition candidate version into candidate testing environment (Staging)
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Staging",
        archive_existing_versions=False
    )
    logger.info(f"Model v{latest_version} moved to Staging.")

    # Strict Quality Control Gate Logic Definition
    QUALITY_GATE = (
            dbcv_score >= 0.03 and
            s_score > 0.01 and
            noise_pct < 20
    )

    # Automated deployment routing decisions based on quality constraints
    if QUALITY_GATE:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )
        logger.info(f"Model v{latest_version} promoted to Production. Performance is stable across density regions.")
    else:
        logger.error(
            f"Quality Gate failed. Reasons: "
            f"{'Low DBCV ' if dbcv_score < 0.03 else ''}"
            f"{'Weak Separation ' if s_score <= 0.01 else ''}"
            f"{'Too much Noise' if noise_pct >= 20 else ''}")

    return run_id
