import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime as dt

logger = logging.getLogger("Data")


def get_processed_data():
    logger.info("Loading data from Excel...")
    df = pd.read_excel(r"D:\ALL Projects\Segmentation\Online Retail.xlsx")
    pd.set_option('display.width', None)
    print(df.head(25))
    logger.info("Dataset Loading Successful...")

    print("=========== Basic Functions ==========")
    logger.info("information about data:")
    print(df.info())

    logger.info("Statistical Operations:")
    print(df.describe().round(2))

    logger.info("Columns of Data:")
    print(df.columns)

    logger.info("number of rows & columns:")
    print(df.shape)

    logger.info("Column types:")
    print(df.dtypes)

    logger.info("=========== Data Cleaning ==========")

    # Validation DT
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%Y%m%d', errors='coerce')

    # Remove Cancelled Invoice (C)
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNo'].str.startswith('C')]

    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    logger.info("Number of Frequency Rows")
    print(df.duplicated().sum()) 
    df = df.drop_duplicates()
    logger.info(f"After Removing Duplicates: {df.shape}")
    print('-------------')

    logger.info("Missing Values")
    print(df.isnull().sum())

    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

    print("=========== Feature Engineering & Aggregation ==========")

    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month
    df['Day'] = df['InvoiceDate'].dt.day
    df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
    df['WeekDay_Name'] = df['InvoiceDate'].dt.weekday

    df['TotalSum'] = df['Quantity'] * df['UnitPrice']

    # Setting a reference date (Snapshot Date)
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    # Data aggregation for each client (The RFM Core)
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

    print(rfm.head(30))
    return df, rfm


if __name__ == "__main__":
    df, rfm = get_processed_data()
