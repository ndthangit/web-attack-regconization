from airflow.sdk import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import os


def create_sample_data():
    """Tạo file dữ liệu mẫu với quyền phù hợp"""
    data_dir = '/opt/airflow/data'
    os.makedirs(data_dir, exist_ok=True)

    # Tạo file input
    input_path = f'{data_dir}/input.txt'
    with open(input_path, 'w') as f:
        for i in range(1, 11):
            f.write(f"Line {i}\n")

    # Thiết lập quyền cho file
    os.chmod(input_path, 0o666)
    os.chmod(data_dir, 0o777)

    print("Sample data created successfully!")


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id="spark_submit_dag_docker",
    default_args=default_args,
    description="DAG để chạy Spark job trên Docker",
    start_date=datetime(2025, 4, 1),
    catchup=False,
) as dag:

    create_data = PythonOperator(
        task_id="create_sample_data",
        python_callable=create_sample_data,
    )
    copy_file_task = BashOperator(
        task_id='copy_input_to_shared',
        bash_command='cp /opt/airflow/data/input.txt /tmp/data/input.txt || echo "File already exists"'
    )

    run_spark = SparkSubmitOperator(
        task_id="run_spark_job",
        application="/opt/airflow/scripts/spark_job.py",
        conn_id="spark_default",
        application_args=["/tmp/data/input.txt"],
        conf={
            "spark.master": "spark://spark-master:7077",
            "spark.executor.memory": "1g",
            "spark.executor.cores": "1",
            "spark.driver.memory": "1g"
        },
        verbose=True,
    )

    process = BashOperator(
        task_id="post_processing",
        bash_command='echo "Spark job completed at $(date)"',
    )

    create_data>> copy_file_task >> run_spark >> process