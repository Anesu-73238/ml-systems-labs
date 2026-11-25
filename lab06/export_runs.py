import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
df = mlflow.search_runs(experiment_names=["lab6-fashion-mnist"], order_by=["start_time ASC"])
cols = ["tags.mlflow.runName", "status"] + [c for c in df.columns
                                            if c.startswith("params.") or c.startswith("metrics.")]
df[cols].to_csv("results/mlflow_runs.csv", index=False)
print(len(df), "runs exported to results/mlflow_runs.csv")