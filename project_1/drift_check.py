import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Split into reference and current data
reference_data = df[:100]
current_data = df[100:]

# Add some artificial drift to current data
current_data = current_data.copy()
current_data["sepal length (cm)"] = current_data["sepal length (cm)"] + 1.5

# Create drift report
report = Report(metrics=[DataDriftPreset()])

snapshot = report.run(
    reference_data=reference_data,
    current_data=current_data
)

# Save report as HTML
snapshot.save_html("drift_report.html")
print("Drift report saved to drift_report.html")