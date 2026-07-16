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
📦 DatasetDataset: Online Retail DatasetDomain: E-commerce transaction analyticsCore fields usedInvoiceNo → transaction identifierCustomerID → customer identifierInvoiceDate → transaction timestampQuantity → purchased unitsUnitPrice → product priceDescription → product nameCountry → customer geographyThe project starts from raw transaction-level retail data and converts it into customer-level behavioral profiles suitable for segmentation.🧹 Data Cleaning & Transaction ProcessingThe first stage focuses on preparing transaction data for trustworthy customer analytics.Cleaning pipelineconverted InvoiceDate to datetimeremoved cancelled invoicesremoved invalid transactions with negative quantity or negative priceremoved duplicatesdropped rows with missing CustomerIDcreated transaction-level revenue feature:PlaintextTotalSum = Quantity × UnitPrice
Final cleaned data541,909 raw transactions~400,000 valid transactions after cleaning~4,000 customers available for segmentationThis stage is important because segmentation quality depends heavily on clean behavioral history, not just the clustering algorithm itself.📊 RFM Feature EngineeringCustomer behavior was modeled using RFM analysis, a classic but highly effective customer analytics framework.RFM definitionMetricDefinitionBusiness meaningRecencyDays since last purchaseHow recently the customer engagedFrequencyNumber of unique purchasesHow often the customer buysMonetaryTotal spendingHow valuable the customer is financiallyCalculation logicFor each customer:PlaintextRecency  = snapshot_date - last_purchase_date
Frequency = number_of_unique_invoices
Monetary = total_customer_spend
Where the snapshot date is defined as:Plaintextsnapshot_date = max(InvoiceDate) + 1 day
📈 Behavioral Correlation Analysis: Pearson vs. SpearmanBefore feeding the engineered RFM metrics into the preprocessing and clustering stages, we conducted a rigorous correlation analysis to map the relationships between customer dimensions.Rather than relying on the standard Pearson Correlation, Spearman Rank Correlation was selected as the core analytical metric.Why Spearman is Mathematically and Commercially Superior Here:Robustness to Extreme Whales (Outliers): Our dataset contains VIP customers with massive spending power (Monetary) and very high transaction counts (Frequency). These outliers represent real business gold, meaning we cannot remove them. Pearson is highly sensitive to extreme raw values and would yield skewed, distorted correlation scores. Spearman converts these raw values to ordinal ranks ($1^{st}, 2^{nd}, 3^{rd} \dots$), neutralizing the impact of massive numerical gaps while fully preserving the underlying order.Capturing Non-Linear Monotonic Relationships: Customer behavior doesn't increase in a strict, uniform linear fashion (e.g., spending exactly $20 more with every extra purchase). Relationships are often exponential or curved. Spearman measures if two variables move in the same direction consistently, regardless of whether that movement forms a straight line.Python# Executed inside inspect_rfm_skew_and_outliers() post-aggregation:
spearman_corr = rfm[['Recency', 'Frequency', 'Monetary']].corr(method='spearman')
sns.heatmap(spearman_corr, annot=True, cmap='viridis', fmt=".2f")
[Optional: Add your Spearman Heatmap plot link or image path here]Business Insight Discovered:Frequency & Monetary yield a high Spearman coefficient (~0.81), confirming that driving consistent purchase habits is the most reliable driver of long-term commercial value, completely unaffected by extreme purchasing spikes.Recency has a negative correlation with both Dimensions, verifying that the longer a customer stays away, the lower their overall transaction volume and financial footprint become.📈 Exploratory Data Analysis & Behavioral InsightsA major goal of the project was not only to cluster customers, but to understand why customer groups differ.Examples of analyses performedRFM distribution analysiscorrelation analysis between Recency, Frequency, and Monetarytop-customer and whale inspectioncountry-level behavioral analysisproduct-level analysis for high-value segmentsmonthly / seasonal activity trendsvisual inspection of customer distributions and segment behavior🔍 Key Data Science InsightsThe analysis produced several strong behavioral findings that shaped the segmentation strategy.1) Frequency is the strongest revenue signalFrequency and Monetary are strongly correlated (0.81), indicating that repeat purchase behavior is the strongest driver of long-term customer value. This suggests that revenue growth depends less on one-time high spend and more on habit formation and repeat purchasing.2) The “51-Day Rule” reveals customer driftMedian recency was 51 days, while the mean was 92 days, revealing a long tail of dormant customers. This became a critical business signal:0–30 days → active zone31–45 days → warning zone46–60 days → danger zone61–100 days → critical zone150+ days → effectively lost customersThis is one of the most actionable outputs of the project because it links segmentation directly to retention timing.3) Not all whales are the sameThe analysis surfaced a VIP paradox:some high-spending customers are true champions with high frequency and low recencyothers are one-time giant spenders who disappeared and are now effectively lostThat distinction matters because “high spend” alone is not enough — customer value must be interpreted together with engagement behavior.4) Customer activity is highly seasonalThe business shows a strong Q4 / November spike, with activity roughly 3× January levels, which has implications for campaign planning, inventory, and staffing.5) Geography and product behavior matterThe analysis highlighted strong geographic concentration in Ireland and the Netherlands, and also uncovered product-line patterns such as the RETROSPOT family behaving as a loyalty anchor across multiple purchases.🔧 Feature Transformation & Preprocessing for ClusteringRaw RFM features were extremely right-skewed, especially Frequency and Monetary, which is dangerous for distance-based clustering.Skewness before transformationRecency: 1.25Frequency: 12.07Monetary: 19.34To stabilize the distributions, the project applied:PlaintextRFM_log = log1p(RFM)
Skewness after transformationRecency: -0.38Frequency: 1.21Monetary: 0.40This preprocessing step is one of the most important technical parts of the project because it makes the RFM space much more suitable for clustering without destroying customer ranking relationships.Outlier treatment philosophyOutliers were inspected using the IQR method, but not removed when they represented legitimate high-value customers. In this project, many extreme points were not noise in the business sense — they were strategically important whale customers.🤖 Why HDBSCAN?ModelHDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)HDBSCAN was chosen over methods like K-Means because customer behavior rarely forms clean spherical clusters.Why it fits this problemdoes not require predefining the number of clusterscan detect clusters of varying densitycan identify noise / outlier customersworks better when customer groups are not uniformly distributedis more suitable for discovering natural behavioral structureThis is especially valuable in customer segmentation because businesses do not want artificial segmentation forced by a fixed K; they want to discover the customer structure that actually exists in the data.🔬 Dimensionality AnalysisTo understand the geometry of the customer space, the project also used dimensionality reduction.PCAPCA preserved ~93% of explained variance, showing that the transformed RFM features capture a strong amount of customer-behavior structure even in lower dimensions.t-SNEt-SNE was used for local structure visualization and helped confirm that customer segments were driven by density patterns, which supports the choice of HDBSCAN over centroid-based clustering.⚙️ Clustering WorkflowHDBSCAN configuration spaceThe project experimented with parameters such as:min_cluster_sizemin_samplesdistance metric (euclidean, manhattan)cluster selection strategyEvaluation metrics usedBecause this is an unsupervised problem, the project did not rely on a single metric. It evaluated clustering quality using:DBCV (Density-Based Cluster Validation)Silhouette ScoreNoise PercentageCluster StabilityThis is an important Data Science point: in segmentation, success is not “one metric.” It is a combination of cluster quality, interpretability, business usefulness, and stability.📊 Final Clustering ResultsThe final HDBSCAN segmentation model discovered 7 natural customer groups.MetricValueDBCV0.0385Silhouette Score0.0936Noise Percentage8.87%Cluster Stability1.3347Segments Discovered7InterpretationPositive DBCV supports the presence of valid density-based structureLow noise percentage suggests that most customers fit into meaningful behavioral groupsCluster stability indicates that the discovered structure is reasonably robustthe final result is not just mathematically acceptable — it is also commercially interpretable through segment profiling🧠 Final Segment ProfilesThe discovered segments were profiled using average RFM values and translated into business-meaningful groups.ClusterSegmentBehavioral Summary5ChampionsMost recent customers, high frequency, high monetary value4Potential LoyalistsStrong customers with good value and room to grow into champions3At RiskPreviously valuable customers beginning to drift away2HibernatingInactive customers with moderate past value1New / Low ValueNew or low-engagement customers with low frequency and low spend0Lost / InactiveLong-inactive customers with minimal recent value-1Outliers / VIP WhalesExceptional customers with very high spend or unusual behaviorRepresentative segment values included:Champions: recency ~23 days, high frequency, high spendPotential Loyalists: recency ~37 days, good purchase depthAt Risk: recency ~52 days — a critical intervention segmentLost / Inactive: recency ~157 days, lowest engagementVIP Outliers: extremely high spenders requiring manual attention💼 Business Value of the SegmentationThe value of the project is not just “we found 7 clusters.”The real value is that each segment maps to different business actions.Examples of strategic actions by segmentChampions → loyalty rewards, VIP treatment, early product accessPotential Loyalists → nudges to increase frequency and move them toward champion statusAt Risk → reactivation campaigns and personalized intervention before they cross into lossNew / Low Value → onboarding and habit-formation campaignsLost / Inactive → low-cost final attempts, then deprioritizationVIP Whales / Outliers → manual review, bespoke service, and dedicated retention treatmentExample strategic findings from the analysis51% of customers (Champions + Potential Loyalists) generate ~80% of revenuethe segmentation supports a 45-day intervention triggerfrequency-building strategies are likely to outperform generic broad campaignswhale customers need different handling depending on whether they are active or dormant🔄 MLflow Lifecycle ManagementThe project includes an MLflow-based workflow to make segmentation experiments reproducible and govern model promotion.MLflow components usedExperiment trackingParameter loggingMetric loggingArtifact loggingPyFunc model packagingModel registrypromotion workflow through quality gatesRegistered modelPlaintextRFM_Segmentation_Production
Example quality gate logicA clustering run can be promoted only if it satisfies quality constraints such as:DBCV ≥ 0.03Silhouette > 0.01Noise % < 20This gives the project a proper MLOps / governance layer, which is uncommon in many segmentation projects and helps differentiate it as a portfolio piece.🌐 FastAPI Segmentation APIThe project exposes the segmentation model through a FastAPI service for real-time customer scoring.Example API responsibilitiesload the registered MLflow modelaccept customer RFM valuesassign cluster label and segment namereturn cluster probability / confidenceflag noise / outlier casesattach business recommendations for the predicted segmentExample prediction inputJSON{
  "Recency": 30,
  "Frequency": 5,
  "Monetary": 1500
}
Example outputJSON{
  "Cluster": 4,
  "Segment_Name": "Potential Loyalists",
  "Probability": 0.87,
  "Is_Noise": false
}
This is a strong part of the project because it shows that the segmentation is not just an offline notebook result — it can be served as a reusable business service.🖥 Streamlit DashboardA Streamlit dashboard is included to make segmentation results usable by analysts or business stakeholders.Dashboard goalsupload customer data for bulk segmentationinspect a single customer profilevisualize customer distribution across segmentsexplore segment-level strategy and recommendationsmonitor prediction history and segment behaviorThis bridges the gap between clustering output and business consumption, which is exactly the kind of thing that makes a Data Science project feel complete rather than purely academic.🛠️ Tech StackLayerTechnologyPurposeSegmentationHDBSCANDensity-based customer clusteringBehavioral ModelingRFM AnalyticsCustomer behavior representationData ProcessingPandas, NumPycleaning, aggregation, transformationPreprocessingScikit-learnscaling, PCA, supporting utilitiesVisualizationMatplotlib, Seaborn, PlotlyEDA and dashboard visualsMLOpsMLflowtracking, registry, lifecycle managementAPIFastAPI + Uvicornsegmentation serviceDashboardStreamlitbusiness-facing analytics interface📁 Project StructureBashproject/
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
Update this section to match your actual repository structure if the filenames differ.🏁 How to Run1) Train / track the segmentation pipelineBashmlflow run .
2) Start the FastAPI serviceBashuvicorn api:app --reload
3) Launch the Streamlit dashboardBashstreamlit run app.py
📌 Why This Project Is Strong from a Data Science PerspectiveThis project demonstrates more than clustering.It shows the full Data Science workflow around segmentation:problem framing in a business contexttransaction cleaning and behavioral feature engineeringdistribution analysis and transformationunsupervised learning with a justified algorithm choicecluster evaluation using multiple quality metricssegment profiling and interpretationtranslation of clusters into concrete business strategyMLflow-based governance and deployment-oriented servingIn other words, this repository is designed to show Data Science maturity, not just model usage.🌱 Future ImprovementsPotential next steps for extending the platform:combine RFM segments with a supervised churn modeladd Customer Lifetime Value (CLV) predictionbuild a next-best-action recommendation layerintegrate the platform with CRM workflowsadd monthly retraining / drift monitoringenrich segmentation with product affinity and category-level behaviorextend the dashboard with segment-level ROI tracking📚 DocumentationThis repository is supported by:Technical Documentation — methodology, RFM design, clustering evaluation, architecture, and deployment workflowBusiness Documentation — segment strategy, retention recommendations, ROI framing, and execution roadmap👨‍💻 AuthorYoussef MahmoudAI / Data Science StudentLinkedIn⭐ Final NoteThis project is not just about grouping customers.It is about turning raw e-commerce transaction data into a segmentation system that supports retention, prioritization, loyalty strategy, and smarter customer investment decisions.It combines behavioral analytics, unsupervised machine learning, model governance, and business interpretation into one end-to-end Data Science project.
