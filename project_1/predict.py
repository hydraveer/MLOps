import mlflow

mlflow.set_tracking_uri("http://localhost:5001")

model = mlflow.sklearn.load_model("models:/IrisClassifier@champion")

#Sample input (iris flower measurements)
# [sepal_length, sepal_width, petal_length, petal_width]
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

labels = {0: "setosa", 1: "versicolor", 2: "virginica"}
print(f"Prediction: {labels[prediction[0]]}")