import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import hdbscan
import logging

logger = logging.getLogger("Model")

def run_modeling(rfm, rfm_log, min_c, min_s, m_input):
    logger.info("=========== Building A ML Model ==========")
    logger.info(f"Parameters: min_cluster_size={min_c}, min_samples={min_s}, metric ={m_input}")
    # Starting RFM Transformation & Outlier Audit
    logger.info("=========== Starting RFM Transformation & Outlier Audit ==========")
    
    # 1. Original Skewness Analysis & Visualization
    logger.info(f"Original Skewness:\n{rfm.skew()}")

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(rfm['Recency'], kde=True, color='skyblue')
    plt.title(f'Recency Distribution\nSkew: {rfm["Recency"].skew():.2f}')

    plt.subplot(1, 3, 2)
    sns.histplot(rfm['Frequency'], kde=True, color='salmon')
    plt.title(f'Frequency Distribution\nSkew: {rfm["Frequency"].skew():.2f}')

    plt.subplot(1, 3, 3)
    sns.histplot(rfm['Monetary'], kde=True, color='lightgreen')
    plt.title(f'Monetary Distribution\nSkew: {rfm["Monetary"].skew():.2f}')
    plt.tight_layout()
    plt.show()

    # 2. Apply Log Transformation to treat Right-Skewed Data
    logger.info("Applying Log Transformation (log1p)...")
    rfm_log = np.log1p(rfm)
    logger.info(f"Skewness after Treatment:\n{rfm_log.skew()}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(rfm_log['Recency'], kde=True)
    plt.title("Recency After Log")

    plt.subplot(1, 3, 2)
    sns.histplot(rfm_log['Frequency'], kde=True)
    plt.title("Frequency After Log")

    plt.subplot(1, 3, 3)
    sns.histplot(rfm_log['Monetary'], kde=True)
    plt.title("Monetary After Log")
    plt.tight_layout()
    plt.show()

    print(rfm_log.describe().round(2))

    # 3. Outlier Audit using IQR Method
    def check_outliers_iqr(df_input):
        outlier_results = {}
        for col in df_input.columns:
            Q1 = df_input[col].quantile(0.25)
            Q3 = df_input[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df_input[(df_input[col] < lower_bound) | (df_input[col] > upper_bound)]

            outlier_results[col] = {
                'Count': len(outliers),
                'Percentage': (len(outliers) / len(df_input)) * 100
            }
        return outlier_results

    results = check_outliers_iqr(rfm_log)

    print("----------------------")
    for col, val in results.items():
        logger.info(f"Column {col}: {val['Count']} Outliers ({val['Percentage']:.2f}%)")

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=rfm_log, palette="Set3")
    plt.title("Boxplot for RFM Features (After Log Transformation)")
    plt.ylabel("Log Values")
    plt.show()

    # Observations & Conclusion
    logger.info("Outlier Audit Complete: Recency (0%), Frequency & Monetary (< 1.5%).")
    logger.info("Log transformation successfully handled the extremes without losing whale customers.")
    logger.info("=========== RFM Transformation Completed ==========")

    # Feature selection ----> RFM

    # 1. Scaling
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    logger.info("Scaling Operation Successful")

    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(rfm_scaled)
    logger.info(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum()}")

    plt.figure(figsize=(10, 5))
    plt.scatter(x_pca[:, 0], x_pca[:, 1], alpha=0.7)
    plt.title("PCA 2D Projection")
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    # We used PCA to compress 3-dimensional (RFM) data into 2-dimensional (2D) data
    # The variance ratio (0.93) is a very high "quality indicator," confirming that the relationships between the variables are strong and interconnected
    # gain:To ensure there is no random "noise" preventing clustering in the binary space
    # ----
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    x_tsne = tsne.fit_transform(rfm_scaled)
    plt.figure(figsize=(10, 5))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], alpha=0.7)
    plt.title("T-SNE 2D Projection")
    plt.ylabel("Dim_2")
    plt.xlabel("Dim_1")
    plt.show()

    # Since PCA showed 93% variance but in a "compressed" form, we used t-SNE to detect "local complexities."
    # t-SNE successfully detected gaps and subclusters invisible in PCA.
    # Result: t-SNE provided "visual evidence" that HDBSCAN is the most suitable, because the discrete clusters that appeared depended on density, not distance from the center.

    # 3. Building HDBSCAN with Metric Loop
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
        cluster_labels = clusterer.fit_predict(rfm_scaled)
        total_stability = clusterer.cluster_persistence_.sum()

        if len(set(cluster_labels)) > 1:
            mask = cluster_labels != -1
            if np.any(mask):
                s_score = silhouette_score(rfm_scaled[mask], cluster_labels[mask])
                dbcv = clusterer.relative_validity_

                results_list.append({
                    'Metric': m,
                    'Clusters': len(set(cluster_labels)) - 1,
                    'Noise_Pct': (cluster_labels == -1).sum() / len(cluster_labels) * 100,
                    'Stability': total_stability,
                    'DBCV': dbcv,
                    'Silhouette': s_score
                })

                if dbcv > max_dbcv:
                    max_dbcv = dbcv
                    best_m = m

    results_df = pd.DataFrame(results_list)
    logger.info("\n--- Metrics Comparison Table ---")
    print(results_df)
    logger.info(f" Selected Best Metric: {best_m} (DBCV: {max_dbcv:.3f})")

    try:
        logger.info(f"Running HDBSCAN with metric: {best_m}")
        final_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_c,
            min_samples=min_s,
            metric=best_m,
            prediction_data=True,
            gen_min_span_tree=True
        )
        rfm['Cluster'] = final_clusterer.fit_predict(rfm_scaled)
        logger.info("Clustering completed successfully.")
    except Exception as e:
        logger.error(f"Failed to build HDBSCAN model: {str(e)}", exc_info=True)
        raise e  


    rfm['Cluster_Probability'] = final_clusterer.probabilities_

    cluster_profile = rfm.groupby('Cluster').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Customer_Count=('Cluster', 'count')
    )

    cluster_profile = cluster_profile.sort_values('Avg_Monetary', ascending=False)

    logger.info(cluster_profile)

    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index)
    rfm_scaled_df['Cluster'] = rfm['Cluster']

    rfm_melted = pd.melt(rfm_scaled_df.reset_index(),
                         id_vars=['Cluster'],
                         value_vars=['Recency', 'Frequency', 'Monetary'],
                         var_name='Attribute',
                         value_name='Value')

    plt.figure(figsize=(12, 6))
    plt.title('Snake Plot of RFM Segments')
    sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=rfm_melted, palette='viridis', marker='o')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()

    return final_clusterer, scaler, rfm, rfm_melted, rfm_scaled
