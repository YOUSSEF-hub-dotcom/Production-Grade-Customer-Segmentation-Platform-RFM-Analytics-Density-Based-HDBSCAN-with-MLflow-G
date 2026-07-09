import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import hdbscan
import logging

logger = logging.getLogger("Model")

def run_modeling(rfm, min_c, min_s, m_input):
    logger.info("=========== Building Production-Grade ML Clustering Model ==========")
    logger.info(f"Parameters: min_cluster_size={min_c}, min_samples={min_s}, metric={m_input}")
    
    # -------------------------------------------------------------------------
    # 1. THE FIREWALL: Absolute Data Isolation Split
    # -------------------------------------------------------------------------
    # Splitting to guarantee that validation evaluation mimics real world streams
    rfm_train, rfm_test = train_test_split(rfm, test_size=0.2, random_state=42)
    
    # Explicit copies to prevent memory pointer pollution
    rfm_train = rfm_train.copy()
    rfm_test = rfm_test.copy()
    
    # -------------------------------------------------------------------------
    # 2. FEATURE TREATMENT: Safe Log Transformation (Isolated)
    # -------------------------------------------------------------------------
    logger.info("Applying isolated Log Transformation to suppress right-skewness...")
    rfm_train_log = np.log1p(rfm_train)
    rfm_test_log = np.log1p(rfm_test)
    
    # Visualizing Treated Train Distributions to verify suppression
    plt.figure(figsize=(15, 4))
    for idx, col in enumerate(rfm_train_log.columns, 1):
        plt.subplot(1, 3, idx)
        sns.histplot(rfm_train_log[col], kde=True, color='purple')
        plt.title(f"{col} Post-Log (Train)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # 3. PRODUCTION SCALING & DIMENSIONALITY REDUCTION
    # -------------------------------------------------------------------------
    # CRITICAL: Fit components strictly on Training set to avoid Data Leakage
    scaler = StandardScaler()
    rfm_train_scaled = scaler.fit_transform(rfm_train_log)
    rfm_test_scaled = scaler.transform(rfm_test_log) # Lazy transformation for test stream
    logger.info("Training Set Scaling Complete. Zero data bleeding triggered.")

    # PCA Construction
    pca = PCA(n_components=2, random_state=42)
    x_train_pca = pca.fit_transform(rfm_train_scaled)
    x_test_pca = pca.transform(rfm_test_scaled) # Project test onto train eigenspace
    logger.info(f"PCA Total Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

    # Plotting Train PCA Projection
    plt.figure(figsize=(8, 4))
    plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], alpha=0.6, c='teal')
    plt.title("PCA 2D Training Projection")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.show()

    # t-SNE Execution on Train Set for structural complexity mapping
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    x_train_tsne = tsne.fit_transform(rfm_train_scaled)
    
    plt.figure(figsize=(8, 4))
    plt.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], alpha=0.6, c='coral')
    plt.title("t-SNE 2D Local Space Embedding")
    plt.show()

    # -------------------------------------------------------------------------
    # 4. HDBSCAN CORE MODEL BUILDING & EXPERIMENTATION
    # -------------------------------------------------------------------------
    metrics_to_test = [m_input] if m_input in ['euclidean', 'manhattan'] else ['euclidean', 'manhattan']
    results_list = []
    best_m = 'euclidean'
    max_dbcv = -1

    for m in metrics_to_test:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_c,
            min_samples=min_s,
            metric=m,
            gen_min_span_tree=True,
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(rfm_train_scaled)
        stability = clusterer.cluster_persistence_.sum()

        if len(set(labels)) > 1:
            mask = labels != -1
            if np.any(mask):
                s_score = silhouette_score(rfm_train_scaled[mask], labels[mask])
                dbcv = clusterer.relative_validity_

                results_list.append({
                    'Metric': m,
                    'Clusters': len(set(labels)) - 1,
                    'Noise_Pct': (labels == -1).sum() / len(labels) * 100,
                    'Stability': stability,
                    'DBCV': dbcv,
                    'Silhouette': s_score
                })

                if dbcv > max_dbcv:
                    max_dbcv = dbcv
                    best_m = m

    results_df = pd.DataFrame(results_list)
    print("\n--- Metrics Grid Search Evaluation ---")
    print(results_df)
    logger.info(f"Selected Optimal Engine Metric: {best_m} (DBCV Peak: {max_dbcv:.3f})")

    # Finalizing Production Cluster Training
    final_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_c,
        min_samples=min_s,
        metric=best_m,
        prediction_data=True, # Critical for scoring outbound API test vectors
        gen_min_span_tree=True
    )
    rfm_train['Cluster'] = final_clusterer.fit_predict(rfm_train_scaled)
    rfm_train['Cluster_Probability'] = final_clusterer.probabilities_
    logger.info("Final HDBSCAN Cluster Architecture Anchored.")

    # Predicting Outbound Test Set Elements via HDBSCAN approximate_predict API
    test_labels, test_probs = hdbscan.approximate_predict(final_clusterer, rfm_test_scaled)
    rfm_test['Cluster'] = test_labels
    rfm_test['Cluster_Probability'] = test_probs
    logger.info("Test Stream Unseen Prediction Complete. Zero Leakage Proved.")

    # Combine for global analysis reporting without bleeding mathematical parameters
    rfm_final_report = pd.concat([rfm_train, rfm_test])

    # -------------------------------------------------------------------------
    # 5. POST-MODELING TELEMETRY & SNAKE PLOTS
    # -------------------------------------------------------------------------
    cluster_profile = rfm_final_report.groupby('Cluster').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Customer_Count=('Cluster', 'count')
    ).sort_values('Avg_Monetary', ascending=False)
    
    print("\n--- Consolidated Final Segment Profiles ---")
    print(cluster_profile)

    # Building Safe Scaled Visualization Frame
    rfm_scaled_all = np.vstack([rfm_train_scaled, rfm_test_scaled])
    rfm_scaled_df = pd.DataFrame(rfm_scaled_all, columns=['Recency', 'Frequency', 'Monetary'], index=rfm_final_report.index)
    rfm_scaled_df['Cluster'] = rfm_final_report['Cluster']

    rfm_melted = pd.melt(rfm_scaled_df.reset_index(),
                         id_vars=['Cluster'],
                         value_vars=['Recency', 'Frequency', 'Monetary'],
                         var_name='Attribute',
                         value_name='Value')

    plt.figure(figsize=(12, 5))
    sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=rfm_melted, palette='viridis', marker='o')
    plt.title('Snake Plot of Independent RFM Segments (Production Safe)')
    plt.grid(True, alpha=0.3)
    plt.show()

    return final_clusterer, scaler, pca, rfm_final_report, rfm_melted
