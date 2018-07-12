import xgboost as xgb
import pandas as pd
import math
import click
from operator import itemgetter


def build_explainer(bst, type="binary", base_score=0.5, lmda=1.0):
    """
    build explain for model on data
    :param bst: model to explain
    :type bst: xgb.Booster
    :param train_data: train_data of the model
    :type train_data: xgb.DMatrix
    :param type:
    :type type: str
    :param base_score:
    :type base_score: float
    :param trees_idx:
    :type trees_idx:
    :return:
    """
    print("Creating the trees of the xgboost model...")
    trees = parse_trees(bst)
    print("Getting the leaf nodes for the training set observations...")
    # train_nodes = bst.predict(train_data, pred_leaf=True)
    print("Building the Explainer...")
    print("STEP 1 of 2")
    tree_list = get_trees_stats(trees, type, base_score, lmda)
    print("STEP 2 of 2")
    # FIXME: time_spend_company is wrong, leaf 0, 115 can use to test
    explainer = build_explainer_from_tree_list(tree_list, bst.feature_names)
    print("DONE!")
    return explainer


def build_explainer_from_tree_list(tree_list, col_names):
    df_left = pd.DataFrame(columns=col_names + ['intercept'], dtype=float)
    df_right = pd.DataFrame(columns=['leaf', 'tree'], dtype=int)
    tree_list_breakdown = pd.concat([df_left, df_right], axis=1)
    # num_trees = len(tree_list)
    print('Getting breakdown for each leaf of each tree...')
    with click.progressbar(tree_list) as bar:
        for tree in bar:
            # print(tree.head())
            tree_breakdown = get_tree_breakdown(tree, col_names)
            tree_breakdown["tree"] = tree.iloc[0]["tree"]
            tree_list_breakdown=tree_list_breakdown.append(tree_breakdown)
    tree_list_breakdown["leaf"] = tree_list_breakdown["leaf"].astype(int)
    tree_list_breakdown["tree"] = tree_list_breakdown["tree"].astype(int)
    return tree_list_breakdown


def get_tree_breakdown(tree, col_names):
    df_left = pd.DataFrame(columns=col_names + ['intercept'], dtype=float)
    df_right = pd.DataFrame(columns=['leaf', 'tree'], dtype=int)
    tree_breakdown = pd.concat([df_left, df_right], axis=1)

    leaves = tree[tree["is_leaf"] == True]["node"]
    for leaf in leaves:
        if int(leaf) == 115:
            pass
        leaf_breakdown = get_leaf_breakdown(tree, leaf, col_names)
        leaf_breakdown["leaf"] = leaf
        tree_breakdown=tree_breakdown.append(leaf_breakdown, ignore_index=True)
    tree_breakdown["leaf"] = tree_breakdown["leaf"].astype(int)
    return tree_breakdown


def get_leaf_breakdown(tree, leaf, col_names):
    """

    :param tree:
    :param tree: pd.DataFrame
    :param leaf:
    :param col_names:
    :return:
    """
    impacts = {}
    path = find_path(tree, leaf)
    reduced_tree = tree[tree["node"].isin(path)][["feature", "uplift_weight"]]
    impacts["intercept"] = reduced_tree.iloc[0]["uplift_weight"]
    reduced_tree["uplift_weight"] = reduced_tree["uplift_weight"].shift(-1)
    tmp = reduced_tree.groupby("feature")["uplift_weight"].sum()
    tmp = tmp[:-1]
    for fname in tmp.index:
        impacts[fname] = tmp[fname]
    return impacts


def find_path(tree, cur_node_id, path=None):
    if path is None:
        path = []
    while cur_node_id > 0:
        path.append(cur_node_id)
        cur_node = tree[tree["node"] == cur_node_id]
        cur_node_id = cur_node["parent"].iloc[0]
    path.append(0)
    path.sort()
    return path


def get_trees_stats(trees_input, type, base_score, lmda):
    """

    :param trees_input:
    :type trees_input: pd.DataFrame
    :param type:
    :param base_score:
    :return:
    """
    trees: pd.DataFrame = trees_input.copy()
    trees['H'] = trees["cover"]
    non_leaf = trees.index[trees["is_leaf"] == False]
    # The default cover (H) seems to lose precision so this loop recalculates
    # it for each node of each tree
    print('Recalculating the cover for each non-leaf...')
    with click.progressbar(reversed(non_leaf), length=non_leaf.shape[0]) as bar:
        for idx in bar:
            left = trees.loc[idx, "yes"]
            right = trees.loc[idx, "no"]
            v = float(trees[trees["id"] == left]["H"])\
                + float(trees[trees["id"] == right]["H"])
            trees = trees.set_value(idx, "H", v)
    if type == "regression":
        base_weight = base_score
    else:
        base_weight = math.log(base_score / (1 - base_score))
    # for leaf only
    trees["weight"] = base_weight + trees["leaf"]
    trees["previous_weight"] = base_weight
    trees = trees.set_value(0, "previous_weight", 0)

    trees["G"] = -trees["weight"] * trees["H"]
    tree_lst = []
    t = 0
    idxs = trees["tree"].unique()

    print("Finding the stats for the xgboost trees...")
    with click.progressbar(idxs) as bar:
        for tree_idx in bar:
            t = t + 1
            cur_tree = trees[trees["tree"] == tree_idx].copy()
            # num_nodes = cur_tree.shape[0]
            non_leaf = cur_tree.index[cur_tree["is_leaf"] == False]
            for idx in reversed(non_leaf):
                left = cur_tree.loc[idx, "yes"]
                right = cur_tree.loc[idx, "no"]
                left_g = float(cur_tree[cur_tree["id"] == left]["G"])
                right_g = float(cur_tree[cur_tree["id"] == right]["G"])

                cur_tree = cur_tree.set_value(idx, "G", left_g + right_g)
                w = -cur_tree.loc[idx, "G"] / (cur_tree.loc[idx, "H"]+lmda)

                cur_tree = cur_tree.set_value(idx, "weight", w)
                left_id = cur_tree[cur_tree["id"] == left].index[0]
                cur_tree = cur_tree.set_value(left_id, "previous_weight", w)
                right_id = cur_tree[cur_tree["id"] == right].index[0]
                cur_tree = cur_tree.set_value(right_id, "previous_weight", w)
            cur_tree["uplift_weight"] = cur_tree["weight"] - cur_tree["previous_weight"]
            tree_lst.append(cur_tree)
    return tree_lst


def parse_trees(bst=None):
    """
    parse tree to pandas dataframe
    :param bst:
    :return:
    :rtype: pd.DataFrame
    """
    bst_str = bst.get_dump(with_stats=True)
    # tree_df = pd.DataFrame(columns=bst.feature_names)
    tree_list = []
    for tree_idx, tree_str in enumerate(bst_str):
        all_nodes_str = list(map(lambda x: x.strip(), tree_str.split("\n")))
        rows_list = []
        parent = {}
        for node_str in all_nodes_str:
            if node_str == "":
                break
            row = {"tree": tree_idx}
            node_lst = node_str.split(":")
            assert len(node_lst) == 2
            # a leaf node
            node_idx = int(node_lst[0])
            row["node"] = node_idx
            is_leaf = node_lst[1].startswith("leaf=")
            row["is_leaf"] = is_leaf
            row["id"] = "{0:d}-{1:d}".format(tree_idx, node_idx)
            if is_leaf:
                for attr_pair in node_lst[1].split(","):
                    key, value = attr_pair.split("=")
                    if key in {"cover", "leaf"}:
                        row[key] = float(value)
            else:
                cond, attrs = node_lst[1].split(" ")
                # remove [ and ]
                cond_key, cond_value = cond[1:-1].split("<")
                row["feature"] = cond_key
                row["split"] = float(cond_value)
                for attr_pair in attrs.split(","):
                    key, value = attr_pair.split("=")
                    if key in {"cover", "gain"}:
                        row[key] = float(value)
                    elif key in {"yes", "no", "missing"}:
                        row[key] = "{0:d}-{1:d}".format(tree_idx, int(value))
                    if key in {"yes", "no"}:
                        parent[int(value)] = node_idx
                        parent[int(value)] = node_idx
            # assert len(rows_list) == node_idx
            rows_list.append(row)
        # assign parent after sort
        # print(parent)
        for item in rows_list:
            if item["node"] > 0:
                item["parent"] = parent[item["node"]]
            else:
                item["parent"] = -1
        rows_list.sort(key=itemgetter("node"))
        # print(tree_idx, len(tree_list))
        # assert len(tree_list) == tree_idx
        tree_list.extend(rows_list)
    df = pd.DataFrame(tree_list,
                      columns=["tree", "node", "id", "parent", "feature",
                               "split", "is_leaf", "yes", "no", "missing",
                               "leaf", "gain", "cover"])
    return df


