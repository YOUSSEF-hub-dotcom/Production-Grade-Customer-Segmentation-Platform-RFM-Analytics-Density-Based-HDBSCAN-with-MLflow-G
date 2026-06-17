import argparse  # بنبدل sys بـ argparse
from data import get_processed_data
from eda import run_eda
from model import run_modeling
from MLflow_LifeCycle import run_mlflow_lifecycle

import logging

logger = logging.getLogger("Main")
from logger_config import setup_logging

setup_logging()


def main():
    logger.info(" Starting RFM Segmentation Pipeline...")

    # --- إعداد الـ Argument Parser ---
    parser = argparse.ArgumentParser(description="RFM Customer Segmentation")

    parser.add_argument("--data_path", type=str, default="D:/ALL Projects/Segmentation/Online Retail.xlsx")
    parser.add_argument("--min_c", type=int, default=100)
    parser.add_argument("--min_s", type=int, default=1)
    parser.add_argument("--m_input", type=str, default="euclidean")

    # الخدعة اللي اتعلمناها عشان يشتغل في الـ Interactive والـ Terminal مع بعض
    args, unknown = parser.parse_known_args()

    # سحب القيم من الـ args
    data_path = args.data_path
    min_c = args.min_c
    min_s = args.min_s
    m_input = args.m_input

    logger.info(f" Configuration: min_c={min_c}, min_s={min_s}, metric={m_input}")

    try:
        logger.info("\n--- Phase 1: Data Processing ---")
        # هنا يفضل تمرر الـ data_path للدالة لو هي بتدعم كدة
        df, rfm, rfm_log = get_processed_data()

        logger.info("\n--- Phase 2: Exploratory Data Analysis ---")
        run_eda(df, rfm, rfm_log)

        logger.info("\n--- Phase 3: Model Building & Optimization ---")
        final_model, scaler, rfm_with_clusters, rfm_melted, rfm_scaled = run_modeling(
            rfm, rfm_log, min_c, min_s, m_input
        )

        logger.info("\n--- Phase 4: MLflow Tracking & Model Governance ---")
        best_m = final_model.metric

        run_id = run_mlflow_lifecycle(
            rfm=rfm_with_clusters,
            rfm_scaled=rfm_scaled,
            rfm_melted=rfm_melted,
            scaler=scaler,
            final_model=final_model,
            min_c=min_c,
            min_s=min_s,
            best_m=best_m,
            m_input=m_input
        )

        logger.info(f"\n Pipeline Completed Successfully! Run ID: {run_id}")
        logger.info(" Open MLflow UI (type 'mlflow ui' in terminal) to see the results.")

    except Exception as e:
        logger.error(f" Pipeline failed: {str(e)}")


if __name__ == "__main__":
    main()
