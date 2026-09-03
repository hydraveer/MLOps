import mlflow
from fastapi import FastAPI
import json
import time
from datetime import datetime
from starlette .responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST



mlflow.set_tracking_uri("http://172.20.10.9:5001")
model = mlflow.sklearn.load_model("models:/IrisClassifier@champion")
app = FastAPI()

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["prediction"]
)

LATENCY = Histogram(
    "prediction_latency_ms",
    "Prediction latency in milliseconds",
)

labels = {0: "setosa", 1: "versicolor", 2: "virginica"}

def log_prediction(features, prediction, latency):
    log = {
        "timestamp": datetime.now().isoformat(),
        "input": features,
        "prediction": prediction,
        "latency_ms": latency
    }
    with open("logs/prediction.log", "a") as f:
        f.write(json.dumps(log) + "\n")


@app.get("/home")
def home():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict")
def predict(features: list[float]):
    start_time = time.time()
    pred= model.predict([features])
    latency = round((time.time()-start_time)*1000,2)
    prediction = labels[pred[0]]
    REQUEST_COUNT.labels(prediction=prediction).inc()
    LATENCY.observe(latency)
    log_prediction(features, prediction, latency)
    return {"prediction": prediction, "latency_ms": latency}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

