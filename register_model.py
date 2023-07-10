import mlflow







def model_registry(best_model,run_params,src_file_path):
    experiment_name="Xgboost_classification"
    run_name="xgboost_fraud_detect"
    print(run_name)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    """
    output_schema = {"prediction": "float"}
    with mlflow.start_run():
        mlflow.sklearn.log_model(best_model, "model")
        if not run_params==None:
            for param in run_params:
                mlflow.log_param(param,run_params[param])
        #mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact(src_file_path, artifact_path="data")
        #mlflow.log_param("input_schema", input_schema)

        # Log output schema
        mlflow.log_param("output_schema", output_schema)
        print("________________________2____________________")
        """


