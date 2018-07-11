import xgboost as xgb


def build_explainer(bst, train_data, type="binary", base_score=0.5,
                    trees_idx=None):
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

    train_nodes = bst.predict(train_data, pred_leaf=True)
    tree_list = parse_trees(bst)
    for tree in tree_list:
        for node in tree:
            if node["is_leaf"]:
                pass
                # g = -1. * node['leaf'] * (node['cover'] + lmda) / eta


def parse_trees(bst=None):
    """
    parse tree to pandas dataframe
    :param bst:
    :return: xgb.Booster
    """
    bst_str = bst.get_dump(with_stats=True)
    # tree_df = pd.DataFrame(columns=bst.feature_names)
    parent = {}
    tree_list = []
    for tree_idx, tree_str in enumerate(bst_str):
        all_nodes_str = list(map(lambda x: x.strip(), tree_str.split("\n")))
        rows_list = []
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
                for attr_pair in node_str.split(","):
                    key, value = attr_pair.split("=")
                    if key in {"cover", "left"}:
                        row[key] = float(value)
            else:
                cond, attrs = node_lst[1].split(" ")
                # remove [ and ]
                cond_key, cond_value = cond[1:-1].split("<")
                row["feature"] = cond_key
                row["split"] = cond_value
                print(cond_key, cond_value)
                for attr_pair in attrs.split(","):
                    key, value = attr_pair.split("=")
                    if key in {"cover", "gain"}:
                        row[key] = float(value)
                    elif key in {"yes", "no", "missing"}:
                        row[key] = int(value)
                    parent[row["yes"]] = node_idx
                    parent[row["no"]] = node_idx
                    assert len(rows_list) == node_idx
                    rows_list.append(row)
        # assign parent
        for key, value in parent.items():
            rows_list[key]["parent"] = value
        assert len(tree_list) == tree_idx
        tree_list.append(rows_list)
    return tree_list

