import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime as dt

logger = logging.getLogger("Data_Pipeline")


def load_data(file_path):
    logger.info("Loading data from Excel...")
    return pd.read_excel(file_path)


def basic_data_overview(df):
    pd.set_option('display.width', None)
    print("=========== Baseline Previews ==========")
    print(df.head(25))
    logger.info("Information about data:")
    df.info()
    logger.info("Statistical Operations:")
    print(df.describe().round(2))
    logger.info(f"Dataset Shape: {df.shape}")


def clean_and_process_data(df):
    logger.info("=========== Starting Data Cleaning ==========")
    
    # Validation Date format
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d', errors='coerce')

    # Remove Cancelled Invoices (C)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]

    # Filter realistic operational bounds
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Handle duplicates
    logger.info(f"Detected Duplicate Rows: {df.duplicated().sum()}") 
    df = df.drop_duplicates()
    
    # Remove rows missing Customer ID
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    logger.info("=========== Feature Engineering & Aggregation ==========")
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['WeekDay_Name'] = df['InvoiceDate'].dt.weekday

    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    # Setting a reference snapshot date
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    # Data aggregation for each client (The RFM Core Matrix)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalSum': 'sum'
    })

    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalSum': 'Monetary'
    }, inplace=True)

    return df, rfm


def inspect_rfm_skew_and_outliers(rfm):
    """
    Performs precise visual and statistical analysis on baseline RFM boundaries 
    before modeling partitions. No structural transformation is applied here.
    """
    logger.info("=========== Advanced Inspection: Baseline Skew & Outliers ==========")
    
    # 1. Log Baseline Skewness
    logger.info(f"\nRaw Matrix Skewness:\n{rfm.skew()}")

    # 2. Extract IQR Outliers Profile
    for col in rfm.columns:
        Q1 = rfm[col].quantile(0.25)
        Q3 = rfm[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = rfm[(rfm[col] < lower_bound) | (rfm[col] > upper_bound)]
        pct = (len(outliers) / len(rfm)) * 100
        logger.info(f"Feature '{col}' -> Potential Outliers: {len(outliers)} ({pct:.2f}%)")

    # 3. Spearman Rank Correlation Analysis (Robust to Outliers)
    logger.info("Executing Spearman Rank Correlation Analysis...")
    spearman_corr = rfm[['Recency', 'Frequency', 'Monetary']].corr(method='spearman')
    logger.info(f"\nSpearman Correlation Matrix:\n{spearman_corr}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(spearman_corr, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
    plt.title("Spearman Rank Correlation Heatmap (RFM Baseline)")
    plt.tight_layout()
    plt.show()
    plt.close()


def run_data_pipeline(file_path):
    """
    Main orchestration flow executing sequential data building blocks.
    """
    logger.info("============ Starting Segment Data Pipeline ============")
    df_raw = load_data(file_path)
    basic_data_overview(df_raw)
    df_clean, rfm_matrix = clean_and_process_data(df_raw)
    
    # Smart inspection step triggered post clean aggregation
    inspect_rfm_skew_and_outliers(rfm_matrix)
    
    logger.info("============ Data Pipeline Completed ============")
    return df_clean, rfm_matrix


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    FILE_PATH = r"D:\ALL Projects\Segmentation\Online Retail.xlsx"
    df, rfm = run_data_pipeline(FILE_PATH)
