from fetchData import *
from feature_selection import *
from config import *
from Evaluation import *
from register_model import *

#src_file_path=src_file_path
file_path=r"dbfs:/FileStore/tables/creditcard.csv"
pandas_df = spark.read.csv(file_path, header=True, inferSchema=True)
df = pandas_df.toPandas()
#df=read_data(src_file_path)
#print(df.head())
X,y=seperate_d_I_values(df)

import mlflow
mlflow.autolog()
experiment_name=r"/testing"
run_name="xgboost_fraud_detect"
print(run_name)
mlflow.set_experiment(experiment_name)
X_train, X_test, y_train, y_test=split_train_test(X,y)

scale_pos_weight=scale_position_weight(y)

xgbc_cv,best_parms=select_best_param(X_train, y_train,scale_pos_weight)
#print(best_parms)



best_xgbc=best_Model(X_train, y_train,scale_pos_weight)

#model_registry(best_model=best_xgbc,run_params=best_parms,src_file_path=src_file_path)

#evaluate(best_xgbc,X_test,y_test)

