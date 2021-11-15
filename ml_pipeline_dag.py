from datetime import datetime, timedelta
from textwrap import dedent
from tasks.ml_pipeline_ import download_data, split_data, preprocess_data, train_model, eval_model
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# These args will get passed on to each operator
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'zli',
    'depends_on_past': False,
    'email': ['zli@beyond.ai'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(31),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'sla_miss_callback': yet_another_function,
    # 'trigger_rule': 'all_success'
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='A simple Machine Learning pipeline',
    schedule_interval=timedelta(days=30),
)

download_data = PythonOperator(
    task_id='download_data',
    provide_context=True,
    python_callable=download_data,
    dag=dag)

split_data = PythonOperator(
    task_id='split_data',
    provide_context=True,
    python_callable=split_data,
    dag=dag)

preprocess_data = PythonOperator(
    task_id='preprocess_data',
    provide_context=True,
    python_callable=preprocess_data,
    dag=dag)

train_model = PythonOperator(
    task_id='train_model',
    provide_context=True,
    python_callable=train_model,
    dag=dag)

eval_model = PythonOperator(
    task_id='eval_model',
    provide_context=True,
    python_callable=eval_model,
    dag=dag)

download_data>>split_data>>preprocess_data>>train_model>>eval_model
# download_data>>split_data
