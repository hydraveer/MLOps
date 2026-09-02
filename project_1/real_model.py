import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("real-model")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with mlflow.start_run():
    params = {"n_estimators": 100, "max_depth": 3}
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "random-forest-model")
    
    print(f"Accuracy: {acc}")