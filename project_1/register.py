import mlflow

mlflow.set_tracking_uri("http://localhost:5001")  # Set the MLflow tracking server URI
run_id = "14f7603062d34bc58146924b8b2192d2"  # Replace with your actual run ID
model__uri = f"runs:/{run_id}/random-forest-model"  # Construct the model URI using the run ID
mlflow.register_model(model__uri, "IrisClassifier")  # Register the model with MLflow

print(f"Model registered successfully with run ID: {run_id}")