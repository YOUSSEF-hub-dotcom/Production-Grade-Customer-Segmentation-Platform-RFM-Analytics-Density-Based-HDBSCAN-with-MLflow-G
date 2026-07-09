import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger("EDA")

def run_eda(df, rfm, rfm_log):
    """
    Executes exploratory data analysis (EDA) to extract behavioral patterns, 
    operational insights, and geometric properties of customer segments.
    """
    logger.info("=========== EDA & Visualization ==========")

    # -------------------------------------------------------------------------
    # 1. Correlation Matrix: Checking Feature Interdependence
    # -------------------------------------------------------------------------
    # Purpose: Evaluate multicollinearity between RFM dimensions on log-scaled features.
    # High correlation might dictate compression needs during downstream modeling.
    logger.info("1. Correlation Matrix between RFM Features")
    correlation = rfm_log.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(correlation, annot=True, cmap='RdYlGn', center=0)
    plt.title("Correlation Heatmap: Recency vs Frequency vs Monetary")
    plt.show()

    # -------------------------------------------------------------------------
    # 2. Pareto Principle Check: Top 10 Revenue Generators
    # -------------------------------------------------------------------------
    # Purpose: Visualize revenue concentration. Helps assess the company's 
    # dependency on high-value "Whale" accounts versus long-tail retail customers.
    logger.info("2. Top 10 Customers by Monetary Value")
    top_10_customers = rfm.nlargest(10, 'Monetary')
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_10_customers.index.astype(str), y=top_10_customers['Monetary'], palette='magma')
    plt.title("Revenue Contribution of Top 10 Customers")
    plt.xlabel("Customer ID")
    plt.show()

    # -------------------------------------------------------------------------
    # 3. Recency Distribution: Uncovering Churn Hazards
    # -------------------------------------------------------------------------
    # Purpose: Inspect the raw timeline footprint. The median line highlights 
    # the general pace of returning clients and flags overall engagement risk.
    logger.info("3. Statistical Summary of Customer Recency")
    plt.figure(figsize=(10, 6))
    sns.histplot(rfm['Recency'], bins=40, kde=True, color='teal')
    plt.axvline(rfm['Recency'].median(), color='red', linestyle='--')
    plt.title("Distribution of Days Since Last Purchase")
    plt.show()

    # -------------------------------------------------------------------------
    # 4. Pairplot: Mapping Geometric Cluster Separability
    # -------------------------------------------------------------------------
    # Purpose: Analyze scatter distributions and density overlaps in 3D-space.
    # Essential for estimating whether density-based clusterers (HDBSCAN) will find distinct structures.
    logger.info("4. Pairplot: Visualizing Natural Groups")
    sns.pairplot(rfm_log, diag_kind='kde', plot_kws={'alpha': 0.4})
    plt.show()

    # Contextual Enrichment: Merging demographic context (Country) for targeted operational views
    df_temp = df.reset_index() if 'CustomerID' not in df.columns else df
    df_unique_customers = df_temp.drop_duplicates(subset=['CustomerID'])[['CustomerID', 'Country']]
    rfm_context = rfm_log.reset_index().merge(df_unique_customers, on='CustomerID', how='left')

    # -------------------------------------------------------------------------
    # 5. Product Preferences: Inventory Alignment for VIP Segments
    # -------------------------------------------------------------------------
    # Purpose: Identify specific stock items favored by top-tier spenders (Top 400).
    # Drives recommendation engines and tailored retention cross-selling strategies.
    logger.info("--- 5. Most Purchased Products by Top Spenders")
    top_customer_ids = rfm.nlargest(400, 'Monetary').index  
    top_products = df[df['CustomerID'].isin(top_customer_ids)]['Description'].value_counts().head(10)

    plt.figure(figsize=(10, 5))
    top_products.plot(kind='barh', color='gold')
    plt.title("Top 10 Products for High-Value Customers")
    plt.show()

    # -------------------------------------------------------------------------
    # 6. Geographic Distribution: Cross-Border Value Profiles
    # -------------------------------------------------------------------------
    # Purpose: Group metrics by country to pinpoint geographical market hubs.
    # Boxplots highlight the purchasing power variance across global borders.
    country_analysis = rfm_context.groupby('Country')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(
        by='Monetary', ascending=False).head(10)
    logger.info(f"--- 6. Average RFM Metrics per Top Countries ---\n{country_analysis.head().to_string()}")

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Country', y='Monetary', data=rfm_context[rfm_context['Country'].isin(country_analysis.index)])
    plt.xticks(rotation=45)
    plt.title("Monetary Distribution (Log) by Top Countries")
    plt.show()

    # -------------------------------------------------------------------------
    # 7. Seasonality & Trendline Analysis: Monthly Volume Pattern
    # -------------------------------------------------------------------------
    # Purpose: Trace operational throughput across time to capture seasonal spikes.
    # Critical for business forecasting, supply-chain budgeting, and marketing campaign timing.
    logger.info("--- 7. Monthly Activity Pattern ---")
    monthly_activity = df.groupby('Month')['InvoiceNo'].nunique()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=monthly_activity.index, y=monthly_activity.values, marker='o', color='red')
    plt.title("Total Transactions (Unique Invoices) per Month")
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 13))
    plt.show()

    # -------------------------------------------------------------------------
    # 8. Weekly Patterns: Day-of-Week Resource Allocation
    # -------------------------------------------------------------------------
    # Purpose: Pinpoint operational "Peak Hours" across the week.
    # Directly informs customer support scheduling and digital infrastructure scaling.
    logger.info("--- 8. Daily Sales Frequency ---")
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df.drop_duplicates('InvoiceNo'), x='DayOfWeek', order=day_order, palette='coolwarm')
    plt.title("Traffic Distribution by Day of Week (Unique Invoices)")
    plt.show()
