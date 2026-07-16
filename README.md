# 📊 Customer Segmentation Platform

An end-to-end **customer segmentation platform** built with **RFM analytics, HDBSCAN clustering, MLflow, FastAPI, and Streamlit** to transform raw e-commerce transactions into **actionable customer segments** for retention, loyalty, and growth strategies.

This project is designed from a **Data Science + ML Engineering** perspective: not just clustering customers, but building a full segmentation workflow that combines **data cleaning, behavioral analytics, unsupervised learning, segment interpretation, model governance, and deployment-oriented serving**.

---

## 📌 Project Overview

In e-commerce, treating all customers the same leads to wasted marketing spend, weak retention strategy, and missed revenue opportunities. A business needs to know:

* who its **highest-value customers** are
* which customers are **drifting toward churn**
* which customers are **worth reactivating**
* which customers are **high spenders but behaviorally unstable**
* where to focus retention, upselling, and loyalty efforts

This project addresses that problem by building a **customer segmentation system** using:

* **RFM (Recency, Frequency, Monetary)** to represent customer behavior
* **HDBSCAN** to discover natural customer segments without predefining the number of clusters
* **MLflow** to track experiments, manage model artifacts, and govern promotion
* **FastAPI + Streamlit** to expose segmentation results through an API and an analytics dashboard

The goal is to turn transaction history into **business-ready segmentation intelligence**.

---

## 🎯 Objectives

This project was built around one central question:

> **Can we segment customers into behaviorally meaningful groups that support targeted marketing, better retention, and more efficient resource allocation?**

### The platform is designed to:

* identify natural customer segments from purchase behavior
* distinguish **champions, potential loyalists, at-risk customers, dormant customers, and whales**
* quantify customer value using **Recency, Frequency, and Monetary behavior**
* surface actionable business insights from clustering results
* provide a reproducible segmentation workflow with **tracking, packaging, and deployment support**

---

## 🧠 End-to-End System Architecture

The project is structured as a full analytics and segmentation workflow:

```text id="d8k2g1"
Raw Transaction Data
        ↓
Data Cleaning & Validation
        ↓
RFM Feature Engineering
        ↓
Behavioral EDA & Insight Discovery
        ↓
Log Transformation + Scaling
        ↓
Dimensionality Analysis (PCA / t-SNE)
        ↓
HDBSCAN Clustering
        ↓
Cluster Evaluation & Segment Profiling
        ↓
MLflow Tracking + Model Registry
        ↓
FastAPI Segmentation API
        ↓
Streamlit Customer Intelligence Dashboard
```

---

## 📦 Dataset

**Dataset:** Online Retail Dataset
**Domain:** E-commerce transaction analytics

### Core fields used

* `InvoiceNo` → transaction identifier
* `CustomerID` → customer identifier
* `InvoiceDate` → transaction timestamp
* `Quantity` → purchased units
* `UnitPrice` → product price
* `Description` → product name
* `Country` → customer geography

The project starts from **raw transaction-level retail data** and converts it into **customer-level behavioral profiles** suitable for segmentation.

---

## 🧹 Data Cleaning & Transaction Processing

The first stage focuses on preparing transaction data for trustworthy customer analytics.

### Cleaning pipeline

* converted `InvoiceDate` to datetime
* removed cancelled invoices
* removed invalid transactions with negative quantity or negative price
* removed duplicates
* dropped rows with missing `CustomerID`
* created transaction-level revenue feature:

```text id="m3s9k4"
TotalSum = Quantity × UnitPrice
```

### Final cleaned data

* **541,909 raw transactions**
* **~400,000 valid transactions after cleaning**
* **~4,000 customers available for segmentation**

This stage is important because segmentation quality depends heavily on **clean behavioral history**, not just the clustering algorithm itself.

---

---

## 📊 RFM Feature Engineering

Customer behavior was modeled using **RFM analysis**, a classic but highly effective customer analytics framework.

### RFM definition

| Metric        | Definition                 | Business meaning                                        |
| ------------- | -------------------------- | ------------------------------------------------------- |
| **Recency** | Days since last purchase   | How recently the customer engaged                       |
| **Frequency** | Number of unique purchases | How often the customer buys                             |
| **Monetary** | Total spending             | How valuable the customer is financially                |

### Calculation logic

For each customer:

```text id="n6r2v7"
Recency  = snapshot_date - last_purchase_date
Frequency = number_of_unique_invoices
Monetary = total_customer_spend
Where the snapshot date is defined as:Plaintextsnapshot_date = max(InvoiceDate) + 1 day
📈 Behavioral Correlation Analysis: Pearson vs. SpearmanBefore feeding the engineered RFM metrics into the preprocessing and clustering stages, we conducted a rigorous correlation analysis to map the relationships between customer dimensions.Rather than relying on the standard Pearson Correlation, Spearman Rank Correlation was selected as the core analytical metric.Why Spearman is Mathematically and Commercially Superior Here:Robustness to Extreme Whales (Outliers): Our dataset contains VIP customers with massive spending power (Monetary) and very high transaction counts (Frequency). These outliers represent real business gold, meaning we cannot remove them. Pearson is highly sensitive to extreme raw values and would yield skewed, distorted correlation scores. Spearman converts these raw values to ordinal ranks ($1^{st}, 2^{nd}, 3^{rd} \dots$), neutralizing the impact of massive numerical gaps while fully preserving the underlying order.Capturing Non-Linear Monotonic Relationships: Customer behavior doesn't increase in a strict, uniform linear fashion (e.g., spending exactly $20 more with every extra purchase). Relationships are often exponential or curved. Spearman measures if two variables move in the same direction consistently, regardless of whether that movement forms a straight line.Python# Executed inside inspect_rfm_skew_and_outliers() post-aggregation:
spearman_corr = rfm[['Recency', 'Frequency', 'Monetary']].corr(method='spearman')
sns.heatmap(spearman_corr, annot=True, cmap='viridis', fmt=".2f")
Business Insight Discovered:Frequency & Monetary yield a high Spearman coefficient (~0.81), confirming that driving consistent purchase habits is the most reliable driver of long-term commercial value, completely unaffected by extreme purchasing spikes.Recency has a negative correlation with both Dimensions, verifying that the longer a customer stays away, the lower their overall transaction volume and financial footprint become.
---

## 📈 Exploratory Data Analysis & Behavioral Insights

A major goal of the project was not only to cluster customers, but to understand **why customer groups differ**.

### Examples of analyses performed

* RFM distribution analysis
* correlation analysis between Recency, Frequency, and Monetary
* top-customer and whale inspection
* country-level behavioral analysis
* product-level analysis for high-value segments
* monthly / seasonal activity trends
* visual inspection of customer distributions and segment behavior

---

## 🔍 Key Data Science Insights

The analysis produced several strong behavioral findings that shaped the segmentation strategy.

### 1) Frequency is the strongest revenue signal

**Frequency and Monetary are strongly correlated (`0.81`)**, indicating that repeat purchase behavior is the strongest driver of long-term customer value. This suggests that revenue growth depends less on one-time high spend and more on **habit formation and repeat purchasing**.

### 2) The “51-Day Rule” reveals customer drift

Median recency was **51 days**, while the mean was **92 days**, revealing a long tail of dormant customers. This became a critical business signal:

* **0–30 days** → active zone
* **31–45 days** → warning zone
* **46–60 days** → danger zone
* **61–100 days** → critical zone
* **150+ days** → effectively lost customers

This is one of the most actionable outputs of the project because it links segmentation directly to retention timing.

### 3) Not all whales are the same

The analysis surfaced a **VIP paradox**:

* some high-spending customers are **true champions** with high frequency and low recency
* others are **one-time giant spenders** who disappeared and are now effectively lost

That distinction matters because “high spend” alone is not enough — customer value must be interpreted together with **engagement behavior**.

### 4) Customer activity is highly seasonal

The business shows a strong **Q4 / November spike**, with activity roughly **3× January levels**, which has implications for campaign planning, inventory, and staffing.

### 5) Geography and product behavior matter

The analysis highlighted strong geographic concentration in **Ireland and the Netherlands**, and also uncovered product-line patterns such as the **RETROSPOT** family behaving as a loyalty anchor across multiple purchases.

---

## 🔧 Feature Transformation & Preprocessing for Clustering

Raw RFM features were **extremely right-skewed**, especially Frequency and Monetary, which is dangerous for distance-based clustering.

### Skewness before transformation

* **Recency:** 1.25
* **Frequency:** 12.07
* **Monetary:** 19.34

To stabilize the distributions, the project applied:

```text id="u7q5n2"
RFM_log = log1p(RFM)
```

### Skewness after transformation

* **Recency:** -0.38
* **Frequency:** 1.21
* **Monetary:** 0.40

This preprocessing step is one of the most important technical parts of the project because it makes the RFM space much more suitable for clustering without destroying customer ranking relationships.

### Outlier treatment philosophy

Outliers were inspected using the IQR method, but **not removed** when they represented legitimate high-value customers. In this project, many extreme points were not noise in the business sense — they were **strategically important whale customers**.

---

## 🤖 Why HDBSCAN?

### Model

**HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**

HDBSCAN was chosen over methods like K-Means because customer behavior rarely forms clean spherical clusters.

### Why it fits this problem

* does **not require predefining the number of clusters**
* can detect **clusters of varying density**
* can identify **noise / outlier customers**
* works better when customer groups are not uniformly distributed
* is more suitable for discovering **natural behavioral structure**

This is especially valuable in customer segmentation because businesses do not want artificial segmentation forced by a fixed `K`; they want to discover the customer structure that actually exists in the data.

---

## 🔬 Dimensionality Analysis

To understand the geometry of the customer space, the project also used dimensionality reduction.

### PCA

PCA preserved **~93% of explained variance**, showing that the transformed RFM features capture a strong amount of customer-behavior structure even in lower dimensions.

### t-SNE

t-SNE was used for local structure visualization and helped confirm that customer segments were driven by **density patterns**, which supports the choice of HDBSCAN over centroid-based clustering.

---

## ⚙️ Clustering Workflow

### HDBSCAN configuration space

The project experimented with parameters such as:

* `min_cluster_size`
* `min_samples`
* distance metric (`euclidean`, `manhattan`)
* cluster selection strategy

### Evaluation metrics used

Because this is an unsupervised problem, the project did not rely on a single metric. It evaluated clustering quality using:

* **DBCV** (Density-Based Cluster Validation)
* **Silhouette Score**
* **Noise Percentage**
* **Cluster Stability**

This is an important Data Science point: in segmentation, success is not “one metric.” It is a combination of **cluster quality, interpretability, business usefulness, and stability**.

---

## 📊 Final Clustering Results

The final HDBSCAN segmentation model discovered **7 natural customer groups**.

| Metric                  |      Value |
| ----------------------- | ---------: |
| **DBCV**                | **0.0385** |
| **Silhouette Score**    | **0.0936** |
| **Noise Percentage**    |  **8.87%** |
| **Cluster Stability**   | **1.3347** |
| **Segments Discovered** |      **7** |

### Interpretation

* **Positive DBCV** supports the presence of valid density-based structure
* **Low noise percentage** suggests that most customers fit into meaningful behavioral groups
* **Cluster stability** indicates that the discovered structure is reasonably robust
* the final result is not just mathematically acceptable — it is also **commercially interpretable** through segment profiling

---

## 🧠 Final Segment Profiles

The discovered segments were profiled using average RFM values and translated into business-meaningful groups.

| Cluster | Segment                   | Behavioral Summary                                               |
| ------- | ------------------------- | ---------------------------------------------------------------- |
| **5**   | **Champions**             | Most recent customers, high frequency, high monetary value       |
| **4**   | **Potential Loyalists**   | Strong customers with good value and room to grow into champions |
| **3**   | **At Risk**               | Previously valuable customers beginning to drift away            |
| **2**   | **Hibernating**           | Inactive customers with moderate past value                      |
| **1**   | **New / Low Value**       | New or low-engagement customers with low frequency and low spend |
| **0**   | **Lost / Inactive**       | Long-inactive customers with minimal recent value                |
| **-1**  | **Outliers / VIP Whales** | Exceptional customers with very high spend or unusual behavior   |

Representative segment values included:

* **Champions:** recency ~23 days, high frequency, high spend
* **Potential Loyalists:** recency ~37 days, good purchase depth
* **At Risk:** recency ~52 days — a critical intervention segment
* **Lost / Inactive:** recency ~157 days, lowest engagement
* **VIP Outliers:** extremely high spenders requiring manual attention

---

## 💼 Business Value of the Segmentation

The value of the project is not just “we found 7 clusters.”
The real value is that each segment maps to **different business actions**.

### Examples of strategic actions by segment

* **Champions** → loyalty rewards, VIP treatment, early product access
* **Potential Loyalists** → nudges to increase frequency and move them toward champion status
* **At Risk** → reactivation campaigns and personalized intervention before they cross into loss
* **New / Low Value** → onboarding and habit-formation campaigns
* **Lost / Inactive** → low-cost final attempts, then deprioritization
* **VIP Whales / Outliers** → manual review, bespoke service, and dedicated retention treatment

### Example strategic findings from the analysis

* **51% of customers (Champions + Potential Loyalists) generate ~80% of revenue**
* the segmentation supports a **45-day intervention trigger**
* frequency-building strategies are likely to outperform generic broad campaigns
* whale customers need **different handling depending on whether they are active or dormant**

---

## 🔄 MLflow Lifecycle Management

The project includes an MLflow-based workflow to make segmentation experiments reproducible and govern model promotion.

### MLflow components used

* **Experiment tracking**
* **Parameter logging**
* **Metric logging**
* **Artifact logging**
* **PyFunc model packaging**
* **Model registry**
* **promotion workflow through quality gates**

### Registered model

```text id="b5t7r1"
RFM_Segmentation_Production
```

### Example quality gate logic

A clustering run can be promoted only if it satisfies quality constraints such as:

* **DBCV ≥ 0.03**
* **Silhouette > 0.01**
* **Noise % < 20**

This gives the project a proper **MLOps / governance layer**, which is uncommon in many segmentation projects and helps differentiate it as a portfolio piece.

---

## 🌐 FastAPI Segmentation API

The project exposes the segmentation model through a FastAPI service for real-time customer scoring.

### Example API responsibilities

* load the registered MLflow model
* accept customer RFM values
* assign cluster label and segment name
* return cluster probability / confidence
* flag noise / outlier cases
* attach business recommendations for the predicted segment

### Example prediction input

```json id="x4n8p2"
{
  "Recency": 30,
  "Frequency": 5,
  "Monetary": 1500
}
```

### Example output

```json id="c9v3k8"
{
  "Cluster": 4,
  "Segment_Name": "Potential Loyalists",
  "Probability": 0.87,
  "Is_Noise": false
}
```

This is a strong part of the project because it shows that the segmentation is not just an offline notebook result — it can be **served as a reusable business service**.

---

## 🖥 Streamlit Dashboard

A Streamlit dashboard is included to make segmentation results usable by analysts or business stakeholders.

### Dashboard goals

* upload customer data for bulk segmentation
* inspect a single customer profile
* visualize customer distribution across segments
* explore segment-level strategy and recommendations
* monitor prediction history and segment behavior

This bridges the gap between clustering output and business consumption, which is exactly the kind of thing that makes a Data Science project feel complete rather than purely academic.

---

## 🛠️ Tech Stack

| Layer               | Technology                      | Purpose                                  |
| ------------------- | ------------------------------- | ---------------------------------------- |
| Segmentation        | **HDBSCAN**                     | Density-based customer clustering        |
| Behavioral Modeling | **RFM Analytics**               | Customer behavior representation         |
| Data Processing     | **Pandas, NumPy**               | cleaning, aggregation, transformation    |
| Preprocessing       | **Scikit-learn**                | scaling, PCA, supporting utilities       |
| Visualization       | **Matplotlib, Seaborn, Plotly** | EDA and dashboard visuals                |
| MLOps               | **MLflow**                      | tracking, registry, lifecycle management |
| API                 | **FastAPI + Uvicorn**           | segmentation service                     |
| Dashboard           | **Streamlit**                   | business-facing analytics interface      |

---

## 📁 Project Structure

```bash id="j2f6q9"
project/
│
├── data.py / data_pipeline.py        # cleaning + transaction preprocessing
├── eda.py                           # exploratory analysis and insight generation
├── model.py                         # HDBSCAN clustering workflow
├── mlflow_lifecycle.py              # experiment tracking + registry workflow
│
├── api.py                           # FastAPI segmentation service
├── app.py                           # Streamlit dashboard
│
├── MLproject                        # MLflow project configuration
├── conda.yaml                       # reproducible environment
├── README.md
└── docs/                            # technical + business documentation
```

> Update this section to match your actual repository structure if the filenames differ.

---

## 🏁 How to Run

### 1) Train / track the segmentation pipeline

```bash id="f3d8m6"
mlflow run .
```

### 2) Start the FastAPI service

```bash id="a7r1w5"
uvicorn api:app --reload
```

### 3) Launch the Streamlit dashboard

```bash id="p6n4t2"
streamlit run app.py
```

---

## 📌 Why This Project Is Strong from a Data Science Perspective

This project demonstrates more than clustering.

It shows the full Data Science workflow around segmentation:

* **problem framing in a business context**
* **transaction cleaning and behavioral feature engineering**
* **distribution analysis and transformation**
* **unsupervised learning with a justified algorithm choice**
* **cluster evaluation using multiple quality metrics**
* **segment profiling and interpretation**
* **translation of clusters into concrete business strategy**
* **MLflow-based governance and deployment-oriented serving**

In other words, this repository is designed to show **Data Science maturity**, not just model usage.

---

## 🌱 Future Improvements

Potential next steps for extending the platform:

* combine RFM segments with a **supervised churn model**
* add **Customer Lifetime Value (CLV) prediction**
* build a **next-best-action recommendation layer**
* integrate the platform with CRM workflows
* add monthly retraining / drift monitoring
* enrich segmentation with product affinity and category-level behavior
* extend the dashboard with segment-level ROI tracking

---

## 📚 Documentation

This repository is supported by:

* **Technical Documentation** — methodology, RFM design, clustering evaluation, architecture, and deployment workflow
* **Business Documentation** — segment strategy, retention recommendations, ROI framing, and execution roadmap

---

## 👨‍💻 Author

**Youssef Mahmoud**
AI / Data Science Student

[LinkedIn](https://www.linkedin.com/in/youssef-mahmoud-63b243361)

---

## ⭐ Final Note

This project is not just about grouping customers.
It is about turning raw e-commerce transaction data into a **segmentation system that supports retention, prioritization, loyalty strategy, and smarter customer investment decisions**.

It combines **behavioral analytics, unsupervised machine learning, model governance, and business interpretation** into one end-to-end Data Science project.
