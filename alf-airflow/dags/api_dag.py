from datetime import datetime

from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.sdk import DAG


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 9, 17),
    'retries': 5

}


with DAG(

    dag_id="api_call_dag",
    default_args=default_args,
    description="DAG lấy data",
    start_date=datetime(2025, 9, 20),
    catchup=False,
) as dag:
    http_call_api = SimpleHttpOperator(
        task_id='http_call_api',
        method='GET',
        endpoint='get',
        headers={"accept": "application/json"},
        log_response=True
    )

