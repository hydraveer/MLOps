# MLOps Learning Notes

## MLflow

### Why MLflow
- Tracks ML experiments like Git tracks code
- Without it: Excel sheets, random folders, no reproducibility
- Flow: Train → Track → Register → Serve → Monitor

### Components
- **Tracking** — logs params, metrics, artifacts per run
- **Model Registry** — version control for trained models
- **Projects** — reproducible training packaging
- **Models** — standard serving format

### What We Did
- Started MLflow server on localhost:5001
- Logged fake run (params + metrics only)
- Logged real RandomForest model on Iris dataset
- Registered model as `IrisClassifier` Version 1
- Set alias `champion` = production-ready model

### Key Concepts
- Run ID = unique ID for each training run (like Git commit)
- Alias `champion` = model currently in production
- Alias `challenger` = new model trying to beat champion
- Overfit = model memorized training data, fails on new data
- 100% accuracy on simple dataset = red flag not a win

### Predict from Registry
- Load model using alias: `models:/IrisClassifier@champion`
- No need for run ID — always loads current champion
- This is what runs inside a FastAPI endpoint in production

## Docker

### Why Docker
- Packages code + dependencies + Python version into one box
- Solves "works on my machine" problem
- Runs identically on laptop, EC2, anywhere

### What We Built
- Dockerized IrisClassifier FastAPI app
- Runs on port 8000
- Loads champion model from MLflow on startup

### Key Gotcha
- Inside Docker, `localhost` = container itself, not your Mac
- `host.docker.internal` didn't work on this setup
- Fix: use Mac's actual IP `*******`
- On EC2 this won't be an issue — MLflow will have its own URL

## Monitoring

### Why Monitoring
- Without it you're blind in production
- Can't know if model is degrading, slow, or broken

### Stack
- prometheus-client → exposes /metrics endpoint
- Prometheus → scrapes /metrics every 15 seconds
- Grafana → reads Prometheus → draws graphs

### Metrics We Track
- prediction_requests_total → count per flower type
- prediction_latency_ms → how fast predictions are

## Retraining Pipeline

### Why
- Manual retraining doesn't scale
- New data arrives → need to retrain automatically
- Never replace a good model with a worse one

### Champion/Challenger Pattern
- Champion = current production model
- Challenger = newly trained model
- Only promote challenger if accuracy > champion

### Key Functions
- get_champion_accuracy() → fetches current champion metrics from MLflow
- train_new_model() → trains with new params, logs to MLflow
- promote_if_better() → compares and promotes if strictly better

### Condition
- Use > not >= in production
- Tying is not a reason to replace a stable model

### Commands
```bash
# Start monitoring stack
docker compose up

# Start iris classifier
docker run -p 8000:8000 -v $(pwd)/logs:/app/logs iris-classifier
```

### URLs
- FastAPI: localhost:8000
- MLflow: localhost:5001
- Prometheus: localhost:9090
- Grafana: localhost:3000