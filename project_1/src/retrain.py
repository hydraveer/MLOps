import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://localhost:5001")  # Set the MLflow tracking server URI


def get_champion_accuracy():
    client = mlflow.MlflowClient()
    champion = client.get_model_version_by_alias("IrisClassifier", "champion")
    run = client.get_run(champion.run_id)
    return float(run.data.metrics["accuracy"])

def train_new_model():
    X,y = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    with mlflow.start_run() as run:
        params = {"n_estimators": 200, "max_depth": 5}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random-forest-model")

        print(f"New model accuracy: {acc}")
        return run.info.run_id, acc

def promote_if_better(run_id, new_acc, champion_acc):
    client = mlflow.MlflowClient()
    if new_acc > champion_acc:
        print(f"New model ({new_acc}) beats champion ({champion_acc}) → promoting")
        model_uri = f"runs:/{run_id}/random-forest-model"
        new_version = mlflow.register_model(model_uri, "IrisClassifier")

        client.set_registered_model_alias("IrisClassifier", "champion", new_version.version)
        print(f"Version {new_version.version} is now champion")
    else:
        # New model is worse or equal → keep existing champion
        print(f"New model ({new_acc}) does not beat champion ({champion_acc}) → keeping existing")

if __name__ == "__main__":
    print("Step 1: Getting champion accuracy...")
    champion_acc = get_champion_accuracy()
    print(f"Champion accuracy: {champion_acc}")

    print("\nStep 2: Training new model...")
    run_id, new_acc = train_new_model()

    print("\nStep 3: Comparing and promoting if better...")
    promote_if_better(run_id, new_acc, champion_acc)




