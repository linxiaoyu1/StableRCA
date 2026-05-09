import networkx as nx
import random
import numpy as np
import torch
import igraph as ig
import torch.nn as nn
from data_scm.edge_functions import LinearFunction, MLPFunction
from data_scm.utils import random_acyclic_orientation
from torch.distributions import Normal, Laplace, Gumbel, Uniform, Exponential


# Generate DAG
def generate_dag(n_nodes, n_edges, graph_type='ER'):
    # Add edges to the DAG
    if graph_type == 'ER':
        G_und = ig.Graph.Erdos_Renyi(n=n_nodes, m=n_edges)
        G = random_acyclic_orientation(G_und)
    elif graph_type == 'SF':
        m = int(round(n_edges / n_nodes))
        # Step 1: undirected Barabasi-Albert graph
        G_und = ig.Graph.Barabasi(n=n_nodes, m=m, directed=False)
        # Step 2: orient edges randomly to avoid cycles
        G = random_acyclic_orientation(G_und)
    else:
        raise NotImplementedError(f"Unknown graph type: {graph_type}")

    return G


def _clone_function_params(function_params, **updates):
    """Return a shallow copy of function parameters without mutating shared args."""
    params = {} if function_params is None else dict(function_params)
    params.update(updates)
    return params


def _make_noise_distribution(noise_type, noise_std):
    """Create a scalar noise distribution with approximately controlled std."""
    if noise_type == 'Gaussian':
        return Normal(0, noise_std)
    elif noise_type == 'Laplace':
        return Laplace(0, noise_std / np.sqrt(2))
    elif noise_type == 'Gumbel':
        return Gumbel(0, np.sqrt(noise_std) * np.sqrt(6) / np.pi)
    elif noise_type == 'Uniform':
        return Uniform(-np.sqrt(3) * noise_std, np.sqrt(3) * noise_std)
    elif noise_type == 'Exponential':
        return Exponential(1.0 / noise_std)
    else:
        raise NotImplementedError(f"Unknown noise type: {noise_type}")


class AssingmentGenerator(nn.Module):
    def __init__(self, function_type, function_params, noise_type, noise_std, aggregate_noise, device):
        """
        Node-level assignment X_j = f_j(Pa_j) + eps_j.

        This preserves the original behavior: one Linear or MLP function maps all
        parents of a node to the child node.
        """
        super(AssingmentGenerator, self).__init__()

        self.function_type = function_type
        self.function_params = _clone_function_params(function_params)
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.aggregate_noise = aggregate_noise
        self.device = device

        if self.function_type == 'Linear':
            self.function = LinearFunction(self.function_params, device)
        elif self.function_type == 'MLP':
            self.function = MLPFunction(self.function_params, device)
        else:
            raise NotImplementedError(f"Unknown function type: {self.function_type}")

        self.noise = _make_noise_distribution(noise_type, noise_std)

    def _sample_noise(self, n_samples):
        noise = self.noise.sample((n_samples,)).to(self.device)
        if self.noise_type == 'Exponential':
            noise = noise - self.noise.mean
        return noise

    def forward(self, x):
        with torch.no_grad():
            function_output = self.function(x.to(self.device))
            noise = self._sample_noise(x.shape[0])
            if self.aggregate_noise == 'additive':
                output = function_output + noise
            else:
                raise NotImplementedError(f"Unknown aggregation type: {self.aggregate_noise}")
        if len(output.shape) == 1:
            return output.unsqueeze(1).float()
        else:
            return output.float()


class HybridAssignmentGenerator(nn.Module):
    def __init__(
        self,
        parents,
        function_params,
        noise_type,
        noise_std,
        aggregate_noise,
        device,
        mlp_prob=0.5,
    ):
        """
        Edge-level heterogeneous assignment:

            X_j = sum_{i in Pa(j)} f_ij(X_i) + eps_j,

        where every edge function f_ij is independently sampled as Linear or MLP.
        This is the recommended synthetic setting for mixed Linear/MLP mechanisms.
        """
        super(HybridAssignmentGenerator, self).__init__()

        self.function_type = 'Hybrid'
        self.function_params = _clone_function_params(function_params)
        self.parents = list(parents)
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.aggregate_noise = aggregate_noise
        self.device = device
        self.mlp_prob = float(mlp_prob)
        self.edge_function_types = {}
        self.edge_functions = nn.ModuleDict()

        for parent in self.parents:
            edge_type = 'MLP' if random.random() < self.mlp_prob else 'Linear'
            edge_params = _clone_function_params(function_params, in_dim=1, out_dim=1)
            self.edge_function_types[parent] = edge_type
            if edge_type == 'Linear':
                self.edge_functions[str(parent)] = LinearFunction(edge_params, device)
            elif edge_type == 'MLP':
                self.edge_functions[str(parent)] = MLPFunction(edge_params, device)
            else:
                raise NotImplementedError(f"Unknown edge function type: {edge_type}")

        self.noise = _make_noise_distribution(noise_type, noise_std)

    def _sample_noise(self, n_samples):
        noise = self.noise.sample((n_samples,)).to(self.device)
        if self.noise_type == 'Exponential':
            noise = noise - self.noise.mean
        return noise

    def forward(self, x):
        with torch.no_grad():
            x = x.to(self.device)
            outputs = []
            for idx, parent in enumerate(self.parents):
                parent_x = x[:, idx:idx + 1]
                outputs.append(self.edge_functions[str(parent)](parent_x))
            function_output = torch.stack(outputs, dim=0).sum(dim=0)
            noise = self._sample_noise(x.shape[0])
            if self.aggregate_noise == 'additive':
                output = function_output + noise
            else:
                raise NotImplementedError(f"Unknown aggregation type: {self.aggregate_noise}")
        if len(output.shape) == 1:
            return output.unsqueeze(1).float()
        else:
            return output.float()


# Generate assignment functions
def generate_assignment(
    G,
    function_type,
    function_params,
    noise_type,
    noise_std,
    aggregate_noise,
    device,
    sample_std=False,
    hybrid_mlp_prob=0.5,
):
    assignment_list = {}
    non_root_nodes = [node for node in G.nodes if G.in_degree(node) != 0]
    for node in non_root_nodes:
        parents = list(G.predecessors(node))
        noise_std_sample = torch.abs(torch.normal(torch.zeros(1), float(noise_std))).to(device) if sample_std else noise_std
        node_params = _clone_function_params(function_params, in_dim=len(parents), out_dim=1)

        if function_type in ['Hybrid', 'hybrid', 'Mixed', 'mixed']:
            assignment = HybridAssignmentGenerator(
                parents=parents,
                function_params=function_params,
                noise_type=noise_type,
                noise_std=noise_std_sample,
                aggregate_noise=aggregate_noise,
                device=device,
                mlp_prob=hybrid_mlp_prob,
            )
        else:
            assignment = AssingmentGenerator(
                function_type,
                node_params,
                noise_type,
                noise_std_sample,
                aggregate_noise,
                device,
            )

        assignment_list[node] = {'parents': parents, 'assignment': assignment}

    return assignment_list


class SCM(torch.nn.Module):
    def __init__(self, args, device):
        super(SCM, self).__init__()
        self.args = args
        for key, value in vars(args).items():
            setattr(self, key.replace('ours_', ''), value)
        self.device = device

        # "auto" preserves the old behavior: one global graph-level function type.
        # Use --ours_function_type Hybrid for edge-level Linear/MLP mixtures.
        if self.function_type == "auto":
            self.function_type = random.choice(["Linear", "MLP"])
        self.hybrid_mlp_prob = getattr(self, "hybrid_mlp_prob", 0.5)

        # Randomly sample noise type
        if self.noise_type == "auto":
            self.noise_type = random.choice(["Gaussian", "Laplace", "Gumbel", "Uniform", "Exponential"])

        self.dag = generate_dag(self.n_nodes, self.n_edges, self.graph_type)
        self.root_nodes = [node for node in self.dag.nodes if self.dag.in_degree(node) == 0]
        self.assignment = generate_assignment(
            self.dag,
            self.function_type,
            self.function_params,
            self.noise_type,
            self.noise_std,
            self.aggregate_noise,
            self.device,
            hybrid_mlp_prob=self.hybrid_mlp_prob,
        )

    def forward(self, n_samples):
        # Sample root nodes
        root_data = (self.max_root - self.min_root) * torch.rand((n_samples, len(self.root_nodes)), device=self.device) + self.min_root
        root_data = root_data.float()

        # Propagate data through the graph
        data = {}
        for i, node in enumerate(nx.topological_sort(self.dag)):
            if node in self.root_nodes:
                data[node] = root_data[:, i].unsqueeze(1)
            else:
                parents_data = torch.cat([data[parent] for parent in self.assignment[node]['parents']], dim=1)
                data[node] = self.assignment[node]['assignment'](parents_data)

        data_values = torch.cat([data[node] for node in self.dag.nodes], dim=1)
        return data_values

    def _build_shifted_assignment(self, node, function_type, function_params, noise_type, noise_std, aggregate_noise):
        parents = self.assignment[node]['parents']
        if function_type in ['Hybrid', 'hybrid', 'Mixed', 'mixed']:
            return HybridAssignmentGenerator(
                parents=parents,
                function_params=function_params,
                noise_type=noise_type,
                noise_std=noise_std,
                aggregate_noise=aggregate_noise,
                device=self.device,
                mlp_prob=self.hybrid_mlp_prob,
            )

        function_params_int = _clone_function_params(function_params, in_dim=len(parents), out_dim=1)
        return AssingmentGenerator(
            function_type,
            function_params_int,
            noise_type,
            noise_std,
            aggregate_noise,
            device=self.device,
        )

    def sample_intervention(self, n_samples, intervention_type, intervention_dict):
        # Sample root nodes
        root_data = (self.max_root - self.min_root) * torch.rand((n_samples, len(self.root_nodes)), device=self.device) + self.min_root
        root_data = root_data.float()

        # Propagate data through the graph
        data = {}
        for i, node in enumerate(nx.topological_sort(self.dag)):
            if node in self.root_nodes:
                if node in intervention_dict.keys():
                    if getattr(self, "verbose", False):
                        print(f"Intervention on node {node} with value {intervention_dict[node]}")
                    if intervention_type == 'hard':
                        data[node] = intervention_dict[node] * torch.ones(n_samples, 1)
                    elif intervention_type == 'soft_function':
                        data[node] = intervention_dict[node].sample((n_samples, 1))
                    elif intervention_type == 'soft_noise':
                        data[node] = intervention_dict[node].sample((n_samples, 1))
                    elif intervention_type == 'soft_distribution':
                        data[node] = intervention_dict[node].sample((n_samples, 1))
                    else:
                        raise NotImplementedError("Unknown intervention type.")
                else:
                    data[node] = root_data[:, i].unsqueeze(1)
            else:
                parents_data = torch.cat([data[parent] for parent in self.assignment[node]['parents']], dim=1)
                if node in intervention_dict.keys():
                    if getattr(self, "verbose", False):
                        print(f"Intervention on node {node} with value {intervention_dict[node]}")
                    if intervention_type == 'hard':
                        data[node] = intervention_dict[node] * torch.ones(n_samples, 1)
                    elif intervention_type == 'soft_function':
                        assignment_obs = self.assignment[node]['assignment']
                        function_type = intervention_dict[node]['function_type']
                        function_params = intervention_dict[node]['function_params']
                        noise_type = assignment_obs.noise_type
                        noise_std = assignment_obs.noise_std
                        aggregate_noise = assignment_obs.aggregate_noise
                        assignment_int = self._build_shifted_assignment(
                            node,
                            function_type,
                            function_params,
                            noise_type,
                            noise_std,
                            aggregate_noise,
                        )
                        data[node] = assignment_int(parents_data)
                    elif intervention_type == 'soft_noise':
                        assignment_obs = self.assignment[node]['assignment']
                        function_type = assignment_obs.function_type
                        function_params = getattr(assignment_obs, 'function_params', self.function_params)
                        noise_type = intervention_dict[node]['noise_type']
                        noise_std = intervention_dict[node]['noise_std']
                        aggregate_noise = assignment_obs.aggregate_noise
                        assignment_int = self._build_shifted_assignment(
                            node,
                            function_type,
                            function_params,
                            noise_type,
                            noise_std,
                            aggregate_noise,
                        )
                        data[node] = assignment_int(parents_data)
                    elif intervention_type == 'soft_distribution':
                        data[node] = intervention_dict[node].sample((n_samples, 1))
                    else:
                        raise NotImplementedError("Unknown intervention type.")
                else:
                    data[node] = self.assignment[node]['assignment'](parents_data)

        int_data_values = torch.cat([data[node] for node in self.dag.nodes], dim=1)
        return int_data_values
