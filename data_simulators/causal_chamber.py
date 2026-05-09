from causalchamber.datasets import Dataset
from typing import Tuple, Dict
import causalchamber
import causalchamber.ground_truth
import networkx as nx
import re
import random
import pandas as pd

CAUSAL_VARIABLES = ['red', 'green', 'blue', 'osr_c', 'v_c', 'current', 'pol_1', 'pol_2',
       'osr_angle_1', 'osr_angle_2', 'v_angle_1', 'v_angle_2', 'angle_1',
       'angle_2', 'ir_1', 'vis_1', 'ir_2', 'vis_2', 'ir_3', 'vis_3', 'l_11',
       'l_12', 'l_21', 'l_22', 'l_31', 'l_32', 'diode_ir_1', 'diode_vis_1',
       'diode_ir_2', 'diode_vis_2', 'diode_ir_3', 'diode_vis_3', 't_ir_1',
       't_vis_1', 't_ir_2', 't_vis_2', 't_ir_3', 't_vis_3']

def subsample_dataset(df: pd.DataFrame, 
                      n: int, 
                      random_state: int | None = None)-> pd.DataFrame:
    if n > len(df):
        raise ValueError(f"n={n} is larger than the number of rows in the DataFrame ({len(df)}).")

    return df.sample(n=n, replace=False, random_state=random_state).reset_index(drop=True) 

def causal_chamber_generator(intervention_name:str, config) -> Dict:
    # Generate the DAG
    true_dag = causalchamber.ground_truth.graph('lt', 'standard')
    G = nx.from_pandas_adjacency(true_dag, create_using=nx.DiGraph)

    dataset = Dataset('lt_interventions_standard_v1', root=config['causal_chamber_data_path'], download=True)
    
    # Generate training dataset
    experiment_training = dataset.get_experiment(name='uniform_reference')
    df_training = experiment_training.as_pandas_dataframe()[CAUSAL_VARIABLES]

    # Generate intervention(anomalous) dataset
    experiment_anomaly = dataset.get_experiment(name=intervention_name)
    df_anomaly = experiment_anomaly.as_pandas_dataframe()[CAUSAL_VARIABLES]

    # Get root cause
    root_cause = intervention_name.replace("uniform_", "")
    root_cause = re.sub(r"_(weak|mid|strong)$", "", root_cause)
    if root_cause.startswith("osr_"):
        parts = root_cause.split("_")
        if parts[-1] in {"1", "2"}:
            return f"osr_angle_{parts[-1]}"

    # subsample the data
    df_training_subsample = subsample_dataset(df_training, n=min(2000, df_training.shape[0]))
    df_anomaly_subsample = subsample_dataset(df_anomaly, n=min(200, df_anomaly.shape[0]))

    # Generate random descendant of root cause as target node
    if root_cause not in G:
        raise ValueError(f"Node '{root_cause}' not found in DAG")
    nodes = {root_cause} | nx.descendants(G, root_cause)
    leaf_nodes = [
        node for node in nodes
        if G.out_degree(node) == 0
    ]

    if not leaf_nodes:
        raise ValueError(
            f"No leaf descendants found for '{leaf_nodes}'"
        )
    target_node = random.choice(leaf_nodes)

    return {
        "graph": G,
        "training_sample": df_training_subsample,
        "anomaly_sample": df_anomaly_subsample,
        "root_cause": root_cause,
        "target_node": target_node
    }    

def draw_graph(G, path):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray', arrows=True)
    plt.savefig(path)
    plt.show()

if __name__ == "__main__":
    import os
    config = {'causal_chamber_data_path':'/home/xiaoyulin/data/causalchamber', 'use_xges_dag':False}
    parameter_list = [#"uniform_red_mid", "uniform_green_mid", "uniform_blue_mid",\
                        #"uniform_red_strong", "uniform_green_strong", "uniform_blue_strong",\
                        #"uniform_pol_1_mid", "uniform_pol_2_mid",\
                        #"uniform_pol_1_strong", "uniform_pol_2_strong",\
                        #"uniform_l_11_mid", "uniform_l_12_mid", "uniform_l_21_mid",\
                        #"uniform_l_22_mid", "uniform_l_31_mid", "uniform_l_32_mid",\
                        #"uniform_diode_vis_1_mid", "uniform_diode_vis_2_mid", "uniform_diode_vis_3_mid",\
                        #"uniform_diode_ir_1_mid", "uniform_diode_ir_2_mid", "uniform_diode_ir_3_mid",\
                        #"uniform_diode_ir_1_strong", "uniform_diode_ir_2_strong", "uniform_diode_ir_3_strong",\
                        #"uniform_t_ir_1_weak", "uniform_t_ir_2_weak", "uniform_t_ir_3_weak",\
                        #"uniform_t_vis_1_weak", "uniform_t_vis_2_weak", "uniform_t_vis_3_weak",\
                        #"uniform_t_ir_1_mid", "uniform_t_ir_2_mid", "uniform_t_ir_3_mid",\
                        #"uniform_t_vis_1_mid", "uniform_t_vis_2_mid", "uniform_t_vis_3_mid",\
                        #"uniform_t_ir_1_strong", "uniform_t_ir_2_strong", "uniform_t_ir_3_strong",\
                        #"uniform_t_vis_1_strong", "uniform_t_vis_2_strong", "uniform_t_vis_3_strong",\
                        #"uniform_osr_1_weak", "uniform_osr_2_weak", "uniform_osr_c_weak",\
                        #"uniform_osr_1_mid", "uniform_osr_2_mid", "uniform_osr_c_mid",\
                        #"uniform_osr_1_strong", "uniform_osr_2_strong", "uniform_osr_c_strong",\
                        #"uniform_v_angle_1_mid", "uniform_v_angle_2_mid", "uniform_v_c_mid",\
                        "uniform_v_angle_1_strong", "uniform_v_angle_2_strong", "uniform_v_c_strong"]

    for parameter in parameter_list:
        data = causal_chamber_generator(parameter, config)
        savefig_dir = '/home/xiaoyulin/projects/RCAWithMissingStructuralKnowledgeCode/dags'
        os.makedirs(savefig_dir, exist_ok=True)
        savefig_path = os.path.join(savefig_dir, 'causalchamber.png')

        draw_graph(data["graph"], savefig_path)
        import pdb; pdb.set_trace()