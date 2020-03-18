FROM python:3.6.9-stretch

RUN pip install mlflow==1.7.0 psycopg2 boto3

EXPOSE 5000

ENTRYPOINT ["mlflow"]
