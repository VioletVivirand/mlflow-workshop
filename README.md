# MLFlow Workshop

## Dependencies

```bash
pip install mlflow scikit-learn pandas numpy scipy boto3
```

## Chapter 1 - Save results locally

The file `train.py` provides a training process by applying [Scikit-learn Elastic Net Model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html), and save the model's metadata and artifact locally.

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
mlflow models serve -m runs:<run_id>/model --no-conda
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
