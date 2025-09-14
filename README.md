# LPECO: Lion with Predictive Error Correction Optimizer

[![Paper](https://img.shields.io/badge/paper-arxiv.XXXX.XXXXX-B31B1B.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Authors:** Siavash Mobarhan, Sogand Tatlari  
**Contact:** sm.steve.moretz@gmail.com, isogandtat@gmail.com  

This repository contains the official implementation and replication package for the paper **"Unlocking Potential: A Case for Neural Network Perseverance on Critical Tabular Tasks Through Principled Optimizer Design."**

## Overview

This paper refutes the current opinion that Gradient Boosted Decision Trees (GBDTs) inherently dominate Neural Networks (NNs) on table-based datasets. We think the performance gap is more because of optimization shortcomings of today, rather than an intrinsic limitation of NNs.

To solve this issue, we propose **LPECO (Lion with Predictive Error Correction)**, a new optimization algorithm that draws inspiration from control theory to enhance the stability of training. Based on our rigorous testing, neural networks optimized by LPECO attain **formal or practical statistical equivalence** with highly-tuned LightGBM and XGBoost models, showing that optimizer design with careful consideration can indeed span the performance gap.

## Installation

The code was tested using Python 3.11.13. To set up the environment:

```bash
# Clone this repository
git clone <repository-url>
cd <repository-directory>

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install the required dependencies
pip install -r requirements.txt
```

## Reproducing the Paper's Results

The entire analysis, such as every comparison of statistics and generating of figures for the paper, can be reproduced by executing a single script.

```bash
python lpeco.py
```

Executing this script will:
1.  Run the full benchmark, comparing LPECO against all other optimizers and GBDT models across the 8 datasets. Results from previous runs are cached in `cached_algorithms_results` to ensure reproducibility and speed up subsequent runs.
2.  Perform all statistical analyses reported in the paper (Friedman test, TOST, Bayesian ROPE analysis, etc.).
3.  Generate all figures from the main paper and the appendices into a local `analysis/plots/` directory.

To force the benchmark to run without using the cache, use the `--no-cache` flag:

```bash
python lpeco.py --no-cache
```

## Key Findings Summary

After executing the script, the test results and numbers derived shall validate the following key results described in the paper:

* **Formal Equivalence:** LPECO achieves formal statistical equivalence with LightGBM. The frequentist test of XGBoost produces test results that are inconclusive, necessitating sophisticated Bayesian analysis and concluding a 98.4% probability of practical equivalence (**Figures 2 & 8** of the paper).

* **Top-Ranked Performance:** On every one of the eight benchmark sets, LPECO achieves the best (lowest) mean rank of 2.625 (**Figure 7** of the paper).

* **Situational Superiority:** On the safety-critical BreastCancer set, LPECO's performance distribution stochastically dominates LightGBM and XGBoost (`p<0.001`) (**Figure 3** of the paper).

## Citation

If you benefit from this work in your own research, you might like to cite our paper:

```bibtex
@article{mobarhan2025lpeco,
  title={Unlocking Potential: A Case for Neural Network Perseverance on Critical Tabular Tasks Through Principled Optimizer Design},
  author={Mobarhan, Siavash and Tatlari, Sogand},
  journal={arXiv preprint},
  year={2025},
  url={https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.