import xgboost as xgb
import numpy as np
import pandas as pd
import xgboostExplainer as xgb_exp


def sigmoid(x):
    return 1/(1+np.exp(-x))


def train():
    """
    train a gbm model for test
    :return: a gbm model
    :rtype: xgb.Booster
    """
    feature_map = ["satisfaction_level", "last_evaluation", "number_project",
                   "average_montly_hours", "time_spend_company",
                   "Work_accident", "promotion_last_5years", "sales", "salary"]
    dtrain = xgb.DMatrix('./train.libsvm', feature_names=feature_map)
    dtest = xgb.DMatrix('./test.libsvm', feature_names=feature_map)
    params = {"objective": "binary:logistic",
              'silent': 1,
              'eval_metric': 'auc',
              'base_score': 0.5,
              "lambda": 1.0}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_iter = 50
    bst = xgb.train(params, dtrain, num_iter, watchlist)
    return bst


# train the model
bst = train()

explainer = xgb_exp.build_explainer(bst, "binary", 0.5, 1.0)

feature_map = ["satisfaction_level", "last_evaluation", "number_project",
               "average_montly_hours", "time_spend_company",
               "Work_accident", "promotion_last_5years", "sales", "salary"]
dtest = xgb.DMatrix('./test.libsvm',
                    feature_names=feature_map).slice([1, 2, 3])
dtest.feature_names = feature_map
pred_breakdown = xgb_exp.explain_prediction(bst, explainer, dtest)

# xgb importance
# xgb.plot_importance(bst)
