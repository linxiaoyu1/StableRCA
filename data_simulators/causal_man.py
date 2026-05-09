from typing import Dict
import os
import pandas as pd
import networkx as nx
import random
import re
# from data_simulators.ours import df_to_xges_digraph

def subsample_dataset(df: pd.DataFrame, 
                      n: int, 
                      random_state: int | None = None)-> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"n={n} is larger than the number of rows in the DataFrame ({len(df)}).")

    return df.sample(n=n, replace=False, random_state=random_state).reset_index(drop=True)   

def causal_man_generator(data_setting:str, config) -> Dict:

    if 'small' in data_setting:
        data_path = os.path.join(config['causal_man_data_path'])

        graph = nx.read_graphml(os.path.join(data_path, 'ground_truth_small.gml'))
        df_training = pd.read_csv(os.path.join(data_path, 'causalmansmall_fully_observable.csv'))
        df_anomaly = pd.read_csv(os.path.join(data_path, f'causalman_{data_setting}.csv'))

        root_cause = re.search(r'do_(.+?)(?:_\d+|_var|$)', data_setting).group(1)

    elif 'medium' in data_setting:
        data_path = os.path.join(config['causal_man_data_path'])

        graph = nx.read_graphml(os.path.join(data_path, 'ground_truth_medium.gml'))
        df_training = pd.read_csv(os.path.join(data_path, 'causalman_medium_observational.csv'))
        df_anomaly = pd.read_csv(os.path.join(data_path, f'causalman_{data_setting}.csv'))

        root_cause = re.search(r'do_(.+?)(?:_\d+|_var|$)', data_setting).group(1)


    # Remove index artifacts early
    df_training = df_training.loc[
        :, ~df_training.columns.str.contains("^Unnamed")
    ]
    df_anomaly = df_anomaly.loc[
        :, ~df_anomaly.columns.str.contains("^Unnamed")
    ]

    graph_nodes = set(graph.nodes)

    training_cols = graph_nodes.intersection(df_training.columns)
    anomaly_cols = graph_nodes.intersection(df_anomaly.columns)

    common_cols = sorted(training_cols.intersection(anomaly_cols))

    if not common_cols:
        raise ValueError(
            "No overlapping columns between data and causal graph nodes."
        )

    df_training_common = df_training[common_cols].copy()
    df_anomaly_common = df_anomaly[common_cols].copy()

    # subsample the data
    df_training_subsample = subsample_dataset(df_training_common, n=min(2000, df_training_common.shape[0]))
    df_anomaly_subsample = subsample_dataset(df_anomaly_common, n=min(200, df_anomaly_common.shape[0]))

    # Generate random descendant of root cause as target node
    if root_cause not in graph:
        raise ValueError(f"Node '{root_cause}' not found in DAG")
    nodes = {root_cause} | nx.descendants(graph, root_cause)
    leaf_nodes = [
        node for node in nodes
        if graph.out_degree(node) == 0
    ]

    if not leaf_nodes:
        raise ValueError(
            f"No leaf descendants found for '{leaf_nodes}'"
        )

    return {
        # "graph": graph if not config['use_xges_dag'] else df_to_xges_digraph(df_training),
        "graph": graph,
        "training_sample": df_training_subsample,
        "anomaly_sample": df_anomaly_subsample,
        "root_cause": root_cause,
        "target_node": 'Sec_C2_Machine1_ProcessResult'
    }   

if __name__ == "__main__":
    import os
    config = {'causal_man_data_path':'/home/xiaoyulin/data/causalman', 'use_xges_dag':False}
    parameter_list = [
        # "small_do_PF_M1_T1_Force_17000", 
        # "small_do_PF_M1_T1_Force_17000_var3000_soft", 
        # "small_do_PF_M1_T1_Fmax_18500", 
        # "small_do_PF_M1_T1_Fmax_18500_var3000_soft",
        # "small_do_PF_M1_T1_sgrad_20", 
        # "small_do_PF_M1_T1_sgrad_20_var4_soft", 
        # "medium_do_PF_M1_T1_Force_17000", 
        # "medium_do_PF_M1_T1_Force_17000_var3000_soft",
        "medium_do_PF_M1_T1_Fmax_18500", 
        "medium_do_PF_M1_T1_Fmax_18500_var3000_soft", 
        "medium_do_PF_M1_T1_sgrad_20", 
        "medium_do_PF_M1_T1_sgrad_20_var4_soft"
        ]

    for parameter in parameter_list:
        data = causal_man_generator(parameter, config)
        print(f'{parameter} data loaded')
        print(f"Root cause {data['root_cause']}")
        import pdb; pdb.set_trace()
        # savefig_dir = '/home/xiaoyulin/projects/RCAWithMissingStructuralKnowledgeCode/dags'
        # os.makedirs(savefig_dir, exist_ok=True)
        # savefig_path = os.path.join(savefig_dir, 'causalchamber.png')

        # draw_graph(data["graph"], savefig_path)
        