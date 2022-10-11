# Zero-shot Transfer Learning within a Heterogeneous Graph via Knowledge Transfer Networks (KTN)

KTN transfers knowledges from label-abundant node types to zero-labeled node types within a sing heterogeneous graph.
More specifically, KTN transfers Heterogeneous Graph Neural Networks (HGNN) that are trained on a source node type to a target node type without using any target labels.

You can see our [NeurIPS 2022 paper](https://arxiv.org/abs/2203.02018) for more details.

## Overview
`Data/` directory contains all files to preprocess OAG-CS raw datasets and extract OAG-ML and OAG-CN subgraphs.
`Model/` directory contains how to train HGNN and KTN models on the preprocessed heterogeneous datasets. 
  
## Setup
This implementation is based on python==3.7. To run the code, you need the dependencies listed in `requirement.txt'

## OAG DataSet
Our current experiments are conducted on Open Academic Graph on Computer Science field (OAG-CS). 
More information to how to download and preprocess OAG-CS dataset can be found in `Data/` directory.

## Usage
Execute `MODEL/run_oag.sh` to run 8 different zero-shot transfer learning tasks on the OAG-CS graph using KTN.
The details of other optional hyperparameters can be found in args.py.

### Citation
Please consider citing the following paper when using our code for your application.

```bibtex
@article{yoon2022zero,
  title={Zero-shot Domain Adaptation of Heterogeneous Graphs via Knowledge Transfer Networks},
  author={Yoon, Minji and Palowitch, John and Zelle, Dustin and Hu, Ziniu and Salakhutdinov, Ruslan and Perozzi, Bryan},
  journal={arXiv preprint arXiv:2203.02018},
  year={2022}
}
```
