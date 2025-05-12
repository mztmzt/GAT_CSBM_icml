# README
This repository contains the experimental code for our paper, "Graph Attention is Not Always Beneficial: A Theoretical Analysis of Graph Attention Mechanisms via CSBMs." (Accepted by International Conference on Machine Learning (ICML) 2025). The `synthetic_experiments` folder includes the code for the synthetic experiments, while the `real_world_experiments` folder contains the code for experiments using three real datasets: Cora, Citeseer, and Pubmed.

To obtain the results for synthetic experiments 1, 2, and 4, run the following command:

```bash
./synthetic_experiments/main_experiments.py
```

For different experiments, you may need to adjust parameters such as `mu`, `sigma`, `p_in`, `p_out`, `T`, and `L` in the code. Please refer to the article for specific settings.

To obtain the results for synthetic experiment 3, run:

```bash
./synthetic_experiments/oversmoothing_experiments.py
```

To get the experimental results for the real datasets mentioned in the main text of the article, run:

```bash
./real_world_experiments/main_experiments.py
```

To access the experimental results for the real datasets included in the appendix of the article, run:

```bash
./real_world_experiments/jmlr_experiments.py
```

## Requirements

The following packages are required to run the code:

```
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.2
scipy==1.10.1
scikit-learn==1.3.2
torch==2.4.0
torchaudio==2.4.0
torchvision==0.19.0
torch-geometric==2.5.3
```

