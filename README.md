# Machine learning for anti-money-laundering detection on AMLworld

Code, notebooks, and result records for the master's thesis

> Mihail Jovanoski (2026). *The Use of Machine Learning in Financial Fraud Detection*.
> Master's thesis, Master in Computational Finance, School of Computing, Union University, Belgrade.
> Supervisor: prof. dr Branko Urošević.

The thesis compares five supervised learning families (penalised logistic regression, a linear
support vector machine, a random forest, histogram-based gradient boosting, and a feed-forward
neural network) for detecting laundering transactions in the HI-Small and LI-Small variants of
IBM AMLworld (Altman et al., 2023). Partitioning, feature construction, imbalance treatment, and
evaluation are held constant, and three nested feature sets separate the contribution of temporal
aggregation (Set B) from that of graph structure (Set C).

## Repository structure

| Path | Content |
| --- | --- |
| `notebooks/01_data_and_eda.ipynb` | Data acquisition, integrity checks, chronological partition, exploratory analysis |
| `notebooks/02_features_and_selection.ipynb` | Feature Sets A, B, and C, causality verification, screening |
| `notebooks/03_models.ipynb` | Hyperparameter search, final fits with five seeds, thresholds, evaluation |
| `notebooks/04_results.ipynb` | Result tables, Friedman/Nemenyi rank procedure, paired bootstrap, typologies |
| `aml/` | Python package used by the notebooks |
| `tests/` | Automated tests on synthetic data |
| `results/` | Integrity, feature, and model records; tables; figures (per-transaction scores excluded) |

## Data

The data are not included. Download the four files `HI-Small_Trans.csv`, `HI-Small_Patterns.txt`,
`LI-Small_Trans.csv`, and `LI-Small_Patterns.txt` from
[Kaggle: IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Community Data License Agreement). Notebook 01 downloads them automatically where possible and
otherwise prints instructions for placing them in `data/raw/`.

## Reproducing the results

The notebooks are written for Google Colaboratory (CPU runtime) with Google Drive mounted, and use
the project folder `MyDrive/aml-thesis/`. Run them in order, each for both variants:

1. `01_data_and_eda.ipynb` (a few minutes)
2. `02_features_and_selection.ipynb` (about 10 minutes)
3. `03_models.ipynb` (about 9 hours in total; the `FAMILIES` parameter allows the sweep to be split
   across sessions, and an interrupted sweep resumes where it stopped)
4. `04_results.ipynb` (about 30 minutes)

Each notebook writes the package code from its own cells, runs the test suite before touching the
real data, and records the installed package versions in `results/`. Setting `DRY_RUN = True`
runs the whole pipeline on a small synthetic dataset.

The tests can also be run locally:

    pip install -r requirements.txt
    pytest -q

## Recorded environment

Google Colaboratory, Python 3.13, 2 virtual CPUs, 12.7 GB RAM; pandas 2.2.3, NumPy 2.4.6,
PyArrow 23.0.1, SciPy 1.16.3, scikit-learn 1.6.1, NetworkX 3.7, TensorFlow 2.21.0, Keras 3.13.2.
The complete lists are in `results/environment*.txt`.

## Reference

Altman, E., Blanuša, J., von Niederhäusern, L., Egressy, B., Anghel, A., & Atasu, K. (2023).
Realistic synthetic financial transactions for anti-money laundering models. *Advances in Neural
Information Processing Systems 36 (Datasets and Benchmarks Track)*.

## License

The code is released under the MIT License (see `LICENSE`). The AMLworld data remain subject to
the license under which they are published on Kaggle.
