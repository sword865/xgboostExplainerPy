import xgboost as xgb
import xgboostExplainer as xgb_exp


feature_map = ["satisfaction_level", "last_evaluation", "number_project",
               "average_montly_hours", "time_spend_company",
               "Work_accident", "promotion_last_5years", "sales", "salary"]

dtrain = xgb.DMatrix('./tests/train.libsvm', feature_names=feature_map)
dtest = xgb.DMatrix('./tests/test.libsvm', feature_names=feature_map)

params = {"objective": "binary:logistic",
          'silent': 1,
          'eval_metric': 'auc',
          'base_score': 0.5,
          "lambda": 1.0}
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_iter = 42
bst = xgb.train(params, dtrain, num_iter, watchlist)

explainer = xgb_exp.build_explainer(bst, "binary", 0.5, 1.0)


sample = xgb.DMatrix('./tests/test.libsvm',
                     feature_names=feature_map).slice([802])
sample.feature_names = feature_map
pred_breakdown = xgb_exp.explain_prediction(bst, explainer, sample)