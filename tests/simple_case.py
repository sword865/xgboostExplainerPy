import xgboost as xgb
import numpy as np
import pandas as pd
# import xgboostExplainer as xgb_exp


def sigmoid(x):
    return 1/(1+np.exp(-x))


def train():
    """
    train a gbm model for test
    :return: a gbm model
    :rtype: xgb.Booster
    """
    dtrain = xgb.DMatrix('./train.libsvm')
    dtest = xgb.DMatrix('./test.libsvm')
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

# xgb importance
xgb.plot_importance(bst)
