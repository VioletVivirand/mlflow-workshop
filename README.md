# MLFlow Workshop

## Dependencies

The Scripts contained in this repository were tested with MLFlow version `1.7.0`:

```bash
pip install mlflow==1.7.0 scikit-learn pandas numpy scipy boto3
```

## Chapter 1 - Track results locally

The file [`train.py`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/train.py) provides a training process by applying [Scikit-learn Elastic Net Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), and save the model's metadata and artifact locally.

### Execute a training job and save the result locally

Execute `train.py` to perform a local training job:

```bash
python train.py
```

The results of experiments will save locally, at `./mlruns` by default

### Perform another training job

Perform another model training job by passing different parameters executing `python train.py <alpha> <l1_ratio>`. e.g.:

```bash
python train.py 0.3 0.7
```

### Host Tracking Server and view the Web Dashboard UI

Host a Tracking Server by CLI commands:

```bash
mlflow server

# or
mlflow ui
```

The Tracking Server will serve the local files at `./mlruns` and runs in the background.

Visit MLFlow Tracking Server's Web Dashboard UI at `http://127.0.0.1:5000`, try to compare the parameters and metrics pairs

### Serve specific model as an Model Inference Server

#### Method 1: Serve with mlflow CLI commands

Pick a local model to serve: Get the `run_id` by viewing the dashboard, then execute with CLI commands `mlflow models serve` :

```bash
mlflow models serve -m runs:/<run_id>/model --no-conda
```

The `--no-conda` option is recommended, so the Inference Server will serve with the dependencies with local resources rather than Anaconda.

> [NOTE]
>
> If the Tracking Server is still running and port `5000` is occupied, try to add `-p` option and assign a different port number

#### Method 2: Build a Docker Image with specific model and Serve the Inference Server as Docker Container

Execute with CLI command `mlflow models build-docker`:

```bash
mlflow models build-docker -m "runs:/<run_id>/model" -n "mlflow-model:latest"
```

Then serve with `docker run`:

```bash
docker run \
  -it -d \
  -p 5000:5000 \
  --name mlflow-model \
  mlflow-model:latest
```

### Make a prediction by sending request to Inference Server

#### Use curl

```bash
curl --location --request POST 'http://127.0.0.1:5005/invocations' \
  --header 'Content-Type: application/json' \
  --header 'format: pandas-split' \
  --data-raw '{"columns":["alcohol","chlorides","citric acid","density","fixed acidity","free sulfur dioxide","pH","residual sugar","sulphates","total sulfur dioxide","volatile acidity"],"data":[[12.8,0.029,0.48,0.98,6.2,29,3.33,1.2,0.39,75,0.66],[12.8,0.029,0.48,0.98,6.2,29,3.33,1.2,0.39,75,0.66]]}'
```

#### Use Python

```python
import requests
import json

url = "http://127.0.0.1:5005/invocations"
payload = "{\"columns\":[\"alcohol\",\"chlorides\",\"citric acid\",\"density\",\"fixed acidity\",\"free sulfur dioxide\",\"pH\",\"residual sugar\",\"sulphates\",\"total sulfur dioxide\",\"volatile acidity\"],\"data\":[[12.8,0.029,0.48,0.98,6.2,29,3.33,1.2,0.39,75,0.66],[12.8,0.029,0.48,0.98,6.2,29,3.33,1.2,0.39,75,0.66]]}"
headers = {
  'Content-Type': 'application/json',
  'format': 'pandas-split'
}

r = requests.request("POST", url, headers=headers, data=payload)
print(json.loads(r.text))
```

## Chapter 2 - Set Remote Tracking Server

The [`train_tracking.py`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/train_tracking.py) add 2 features:

1. Set remote Tracking Server, and save metadata and artifacts into remote environment's filesystem
1. Set the name of experiment to separate different projects

### Set remote Tracking Server

In Python, we can set remote Tracking Server by using `mlflow.set_tracking_uri()` function, refers to [line 30](https://github.com/VioletVivirand/mlflow_workshop/blob/b6e7d7745d2dd27be1e5b72329c8d9c1e778e3d9/train_tracking.py#L30):

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI",
                                  "http://127.0.0.1:5000"))
```

So set Tracking Server URI by assigning environment `MLFLOW_TRACKING_URI`:

```bash
export MLFLOW_TRACKING_URI=<URI>

# Verify
echo $MLFLOW_TRACKING_URI
```

And if the environment variable is unset, it will assign `http://127.0.0.1:5000` as Tracking Server URI.

### Set the name of experiment

In order to separate different projects, use `mlflow.set_experiment() function`, refers to [line 32](https://github.com/VioletVivirand/mlflow_workshop/blob/b6e7d7745d2dd27be1e5b72329c8d9c1e778e3d9/train_tracking.py#L34):

```python
mlflow.set_experiment("Elastic Net")
```

### Perform a Training Job

Start a Tracking Server first:

```bash
mlflow server
```

Now we can visit the Tracking Server Web UI dashboard, it runs with no record found.

Execute `train_tracking.py` to perform a training job:

```bash
python train_tracking.py
```

Then refresh the webpage, the training result shows up immediately.

> [NOTE] The artifact may not be uploaded (but no error returns from Tracking Server) for unknown reason. Try the method in Chapter 3.

## Chapter 3 - Host MLFlow, PostgreSQL Database, MinIO Stack

The complete tracking platform has three components:

* MLFlow Tracking Server
* [PostgreSQL Database](https://www.postgresql.org): A popular open-source relational database, can be used to store the models' metadata
* [MinIO](https://min.io): Amazon S3-compatible Object Storage, used to store the artifacts of the models

### Start PostgreSQL and MinIO

With [`docker-compose.yml`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/docker-compose.yml) file, web can run the services with [Docker Compose](https://docs.docker.com/compose/):

```bash
# Prepare local directories as bind-mount persistent volumes
mkdir -p ./data/minio/mlflow
mkdir -p ./data/postgres

# Launch services
docker-compose up -d
```

> [NOTE]
>
> We don't run MLFlow Tracking Server along with other components due to the lack of the Healthcheck mechanism. The MLFlow Tracking Server should be initialized after the Database created, so without healthchecking, we can't use `depends_on` keyword to assign the right order to run the services and causes the failure.

Use `docker-compose ps` commands to verify if the services become stable.

### Start MLFlow Tracking Server

Build a MLFlow Docker Image, contains:

* `psycopg2`: PostgreSQL Database Driver
* `boto3`: AWS Python SDK

Build with `docker build` commands with [`Dockerfile`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/Dockerfile) file:

```bash
docker build -t mlflow:v1.7.0 .
```

Then run as Docker Container:

```bash
docker run -it -d \
  --network="mlflow_net" \
  -e MLFLOW_S3_ENDPOINT_URL=http://minio:9000 \
  -e AWS_ACCESS_KEY_ID=minioadmin \
  -e AWS_SECRET_ACCESS_KEY=minioadmin \
  -p 5000:5000 \
  --name mlflow \
  mlflow:v1.7.0 \
  server --host="0.0.0.0" \
    --backend-store-uri="postgresql+psycopg2://mlflow:mlflow@postgres/mlflow" \
    --default-artifact-root="s3://mlflow"
```

### Track with PostgreSQL Database and MinIO Object Storage

[`train_tracking_s3_psql.py`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/train_tracking_s3_psql.py) add some new features:

* On [line 22](https://github.com/VioletVivirand/mlflow_workshop/blob/b6e7d7745d2dd27be1e5b72329c8d9c1e778e3d9/train_tracking_s3_psql.py#L22), set environment variables to redirect the S3 URL to out self-hosted MinIO object storage
* On [line 23](https://github.com/VioletVivirand/mlflow_workshop/blob/b6e7d7745d2dd27be1e5b72329c8d9c1e778e3d9/train_tracking_s3_psql.py#L23) and [line 24](https://github.com/VioletVivirand/mlflow_workshop/blob/b6e7d7745d2dd27be1e5b72329c8d9c1e778e3d9/train_tracking_s3_psql.py#L24), set S3 credentials

So simply execute the script:

```python
python train_tracking_s3_psql.py
```

Now, when we log the results with MLFlow Tracking Server, the metadata are saved into PostgreSQL Database, and model artifacts are saved into MinIO object storage automatically.

### Serve Model with Object Storage

Visit MLFlow Tracking Server and view the specific model detail page to get the URI of the model, the format should be like `s3://bucket/<experiment_id>/<run_id>/artifacts/model`. Then export environment variables and server the model with commands:

```bash
export MLFLOW_S3_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

mlflow models serve -m s3://mlflow/<experiment_id>/<run_id>/artifacts/model --no-conda
```

> [NOTE]
>
> If the Tracking Server is still running and port `5000` is occupied, try to add `-p` option and assign a different port number

## One More Thing - Use H2O AutoML with MLFlow Tracking Server

[`H2OAutoML.ipynb`](https://github.com/VioletVivirand/mlflow_workshop/blob/master/H2OAutoML.ipynb) Illustates the step-by-step guide to perform an AutoML Training Job with MLFlow Tracking Server, enjoy it!
