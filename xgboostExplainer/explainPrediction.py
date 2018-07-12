import xgboost as xgb
import pandas as pd
import numpy as np
import click


def explain_prediction(bst, explainer, data):
    """

    :param bst:
    :type bst: xgb.Booster
    :param explainer:
    :type explainer: pd.DataFrame
    :param data:
    :return:
    """
    nodes = bst.predict(data, pred_leaf=True)
    colnames = list(explainer.columns.values)[:-2]

    preds_breakdown = pd.DataFrame(np.zeros((nodes.shape[0], len(colnames))),
                                   columns=colnames)

    print("Extracting the breakdown of each prediction...")
    num_trees = nodes.shape[1]
    with click.progressbar(range(num_trees), num_trees) as bar:
        for idx in bar:
            nodes_for_tree = nodes[:, idx]
            tree_breakdown = explainer[explainer["tree"] == idx].fillna(0)
            preds_breakdown_for_tree = tree_breakdown.loc[
                pd.match(nodes_for_tree, tree_breakdown["leaf"])][colnames] \
                .reset_index(drop=True)
            preds_breakdown = preds_breakdown + preds_breakdown_for_tree
    print("DONE!")
    return preds_breakdown

