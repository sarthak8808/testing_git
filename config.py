src_file_path=r"C:\Users\PatkiSarthakShridhar\Projects\MlopsProject\s3\creditcard.csv"

xgb_grid = {'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5],
                'gamma': [0],
                'subsample': [0.8],
                'colsample_bytree': [0.8] }

config_experiment_name="Xgboost_classification"
config_run_name="xgboost_fraud_detect"