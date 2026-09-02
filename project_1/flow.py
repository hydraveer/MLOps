import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("test1")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)
    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("loss", 0.34)
    print("Run logged!")
