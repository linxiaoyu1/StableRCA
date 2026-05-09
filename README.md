# RCA Baselines

This repository provides implementations and experiment scripts for RCA benchmark evaluation on synthetic and real-world datasets.

## Environment

Create and activate the conda environment:

```bash
cd StableRCA
conda env create -f environment.yml
conda activate stable_rca_baselines
```

Install the PyRCA dependency by cloning the repository and installing it locally:

```bash
git clone git@github.com:salesforce/PyRCA.git
cd PyRCA
pip install .
```

If you encounter the following `ValueError` when running `main.py`:

```bash
ValueError: numpy.dtype size changed, may indicate binary incompatibility.
Expected 96 from C header, got 88 from PyObject
```

force reinstall `numpy` and `scikit-learn`:

```bash
conda install --force-reinstall numpy scikit-learn
```

---

## Running Experiments

### Synthetic Data

Run experiments on synthetic data with:

```bash
bash exp_synthetic_data.sh
```

The default configuration in `exp_synthetic_data.sh` uses a synthetic graph with 50 nodes and 100 edges, and evaluates methods using the true graph.

To evaluate methods using an XGES-discovered graph, set:

```bash
--graph_mode xges
```

To evaluate methods using a corrupted graph, set:

```bash
--graph_mode corrupted
```

For corrupted graphs, the corruption level is controlled by the following parameters:

```bash
--corrupt_delete_frac
--corrupt_reverse_frac
--corrupt_add_frac
```

In the experiments reported in the paper, we use the following corruption settings:

```text
30% corruption: --corrupt_delete_frac 0.15 --corrupt_reverse_frac 0.10 --corrupt_add_frac 0.05
50% corruption: --corrupt_delete_frac 0.20 --corrupt_reverse_frac 0.20 --corrupt_add_frac 0.10
70% corruption: --corrupt_delete_frac 0.25 --corrupt_reverse_frac 0.25 --corrupt_add_frac 0.20
```

To change the number of graph nodes, use:

```bash
--n_nodes
```

To change the number of intervention nodes, use:

```bash
--n_intervention_nodes
```

### Real-World Datasets

Run experiments on a real-world dataset with:

```bash
bash exp_[DATASET_NAME].sh
```

Replace `[DATASET_NAME]` with the corresponding dataset name.

For example:

```bash
bash exp_prorca.sh
bash exp_sockshop.sh
bash exp_causalman.sh
bash exp_causalchamber.sh
bash exp_rcaeval.sh
```

---

## Notes

- Synthetic experiments support true graphs, XGES-discovered graphs, and manually corrupted graphs.
- The graph corruption setting only changes the graph provided to graph-based RCA methods. The data are still generated from the original ground-truth SCM.
- For multiple-intervention experiments, set `--n_intervention_nodes` to the desired number of root causes.
