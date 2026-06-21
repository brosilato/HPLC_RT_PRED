<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#project-description">Project Description</a>
      <ul>
        <li><a href="#built-with">Built with</a></li>
      </ul>
    </li>
  </ol>
</details>

<!-- PREOJECT DESCRIPTION -->
## Project Description

This project aims to build a deep learning solutions to predict HPLC retention times. Three architectures will be considered:

1.  Multi-Layer Perceptron (MLP). I will use fingerprints as features.
2.  Convolutional Neural Network (CNN). I will use tokenized smiles as features.
3.  Graph neural network (GNN). It will use node features but their nature is still undecided.

### Built with

The fundamentals libraries that I employed building this project are:

* RDKit
* Scikit-Learn
* PyTorch

<!-- METLIN SMRT DATASET -->
## The METLIN SMRT Dataset

The METLIN small molecule retention times (SMRT) was downloaded as a csv file on 4/20/2025 from 
[FigShare](https://figshare.com/articles/dataset/The_METLIN_small_molecule_dataset_for_machine_learning-based_retention_time_prediction/8038913).

The file includes the PubMed ID of each compound, the measured retention time, and International Chemical Identifier (InChI).

