from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from config import *
import xgboost as xgb

xgb_model = xgb.XGBRegressor(tree_method='cpu')
config_xgb_grid=xgb_grid
def seperate_d_I_values(df):
    X = df.iloc[:, 1:-1].values
    y = df['Class'].values
    return X,y

def split_train_test(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
    return X_train, X_test, y_train, y_test

def scale_position_weight(y): 
    scale_pos_weight= y.shape[0] / y.sum() - 1
    return scale_pos_weight
def select_best_param(X_train, y_train,scale_pos_weight):
    
  
    param = {
             'learning_rate': 0.1,
            'verbosity': 2,
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'scale_pos_weight': scale_pos_weight,
            'n_estimators': 300}
    
    param['tree_method'] = 'hist'
    xgb_grid = config_xgb_grid
    xgbc = XGBClassifier(**param)
    xgbc_cv = GridSearchCV(estimator = xgbc, param_grid = xgb_grid, cv = 3, scoring = 'average_precision', n_jobs = -1, verbose = 2)
    xgbc_cv.fit(X_train, y_train)
    print('Best parameters: ', xgbc_cv.best_params_)
    print('Best score: ', xgbc_cv.best_score_)
    best_parms=xgbc_cv.best_params_
    return xgbc_cv,best_parms

def best_Model(X_train, y_train,scale_pos_weight):
    best_xgbc = XGBClassifier(scale_pos_weight = scale_pos_weight,
                          objective = 'binary:logistic',
                          tree_method = 'hist',
                          max_depth = 6,
                          min_child_weight = 1,
                          gamma = 0,
                          subsample = 0.6,
                          colsample_bytree = 0.6,
                          alpha = 0,
                          learning_rate = 0.01,
                          n_estimators = 2000)

    best_xgbc.fit(X_train, y_train)
    y_pred = best_xgbc.predict(X_train)
    print(y_pred)
    return best_xgbc