import glob

from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sdk import DAG
import os
import json
import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.mongo.hooks.mongo import MongoHook


from datetime import datetime
import logging

from bson import ObjectId

# Thư mục lưu file JSON riêng cho mỗi collection
EXPORT_DIR = "/tmp/mongo_export"
CSV_EXPORT_DIR = "/tmp/csv_export"

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 9, 17),
    'retries': 5
}


def extract_data_from_mongodb(**kwargs):
    hook = MongoHook(mongo_conn_id='mongo_default', ssl=False)
    client = hook.get_conn()
    db = client["web-attack-db"]

    # Create the EXPORT_DIR if it doesn't exist
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)

    for coll_name in db.list_collection_names():
        logging.info(f"Extracting collection: {coll_name}")
        cursor = db[coll_name].find({})
        docs = list(cursor)

        if not docs:
            logging.info(f"Collection {coll_name} is empty, skipping.")
            continue

        # Process data to string if ObjectId exists
        def _to_str(o):
            if isinstance(o, ObjectId):
                return str(o)
            if isinstance(o, list):
                return [_to_str(x) for x in o]
            if isinstance(o, dict):
                return {k: _to_str(v) for k, v in o.items()}
            return o

        docs = [_to_str(d) for d in docs]

        # Save the data into a JSON file in EXPORT_DIR
        coll_path = os.path.join(EXPORT_DIR, f"{coll_name}.json")
        logging.info(f"Writing to local file: {coll_path}")
        with open(coll_path, 'w') as f:
            json.dump(docs, f, indent=4)


def upload_files_to_minio(**kwargs):
    s3_hook = S3Hook(aws_conn_id='minio_default')
    export_bucket = 'aminer'
    export_dir = EXPORT_DIR  # Directory where the files are saved

    # Iterate over the files in the export directory
    for file_name in os.listdir(export_dir):
        file_path = os.path.join(export_dir, file_name)

        if os.path.isfile(file_path):
            logging.info(f"Uploading {file_name} to MinIO bucket {export_bucket}")

            # Upload the file to MinIO
            s3_hook.load_file(filename=file_path, key=file_name, bucket_name=export_bucket, replace=True)
            logging.info(f"Uploaded {file_name} to {export_bucket}")

def load_csv_to_postgres(**kwargs):
    """
    Đọc toàn bộ CSV trong /tmp/csv_export do Spark sinh ra,
    xoá dữ liệu cũ theo source_file rồi append vào Postgres.
    """
    # Airflow Connection ID (đặt trong UI): ví dụ 'postgres_default' hoặc 'analytics_postgres'
    pg_conn_id = os.getenv("POSTGRES_CONN_ID", "postgres_default")
    hook = PostgresHook(postgres_conn_id=pg_conn_id)

    schema = os.getenv("POSTGRES_SCHEMA", "public")
    table = os.getenv("POSTGRES_TABLE", "pretrain_features")
    fqtn = f'{schema}.{table}'

    # Lấy list CSV
    csv_files = sorted(glob.glob(os.path.join(CSV_EXPORT_DIR, "*.csv")))
    if not csv_files:
        logging.info(f"No CSV found in {CSV_EXPORT_DIR}, skip load.")
        return

    # Nạp lần lượt từng file để giữ tính idempotent theo source_file
    engine = hook.get_sqlalchemy_engine()
    with engine.begin() as conn:
        for csv_path in csv_files:
            logging.info(f"Loading CSV -> Postgres: {csv_path}")
            df = pd.read_csv(csv_path)

            # Ghi mới
            df.to_sql(name=table, con=conn, schema=schema, if_exists='replace', index=False)

    logging.info("Postgres load completed.")


with DAG(
        dag_id='batch_ingestion_dag',
        default_args=default_args,
        catchup=False
) as dag:
    extract_task = PythonOperator(
        task_id='extract_from_mongodb',
        python_callable=extract_data_from_mongodb
    )

    upload_task = PythonOperator(
        task_id='upload_to_minio',
        python_callable=upload_files_to_minio
    )


    run_spark = SparkSubmitOperator(
        task_id="run_spark_batch_job",
        application="/opt/airflow/scripts/spark_batch_job.py",
        conn_id="spark_default",
        application_args=[],
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.executor.memory": "1g",
            "spark.executor.cores": "1",
            "spark.driver.memory": "1g"
        },
        verbose=True,
    )

    load_to_postgres = PythonOperator(  # NEW
        task_id='load_to_postgres',
        python_callable=load_csv_to_postgres
    )

    # Luồng xử lý: Mongo -> GE -> Spark(ghi Postgres)
    extract_task >> upload_task >> run_spark >> load_to_postgres