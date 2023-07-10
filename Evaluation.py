from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt



def evaluate(best_xgbc,X_test,y_test):
    y_pred_proba_xgb = best_xgbc.predict_proba(X_test)[:, 1]
    precision_xgb, recall_xgb, threshold_xgb = precision_recall_curve(y_test, y_pred_proba_xgb)
    plt.plot(recall_xgb, precision_xgb, label = 'XGBoost (PRAUC = {:.3f})'.format(auc(recall_xgb, precision_xgb)))
    plt.title('The Precison-Recall Curve of the XGBoost model')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc = 'lower left')
    plt.show()