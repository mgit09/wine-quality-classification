# Wine Quality Prediction

Final project for the Machine Learning course (2026/2) at the Federal University of Minas Gerais (UFMG).

This project investigates wine quality (red and white wines) based on 11 physicochemical properties, applying exploratory data analysis, discriminant analysis, resampling techniques, Principal Component Analysis (PCA), and comparison of machine learning models.


## Status

The analysis pipeline (steps `s1` to `s10`) is complete: exploratory analysis, discriminant analysis, PCA, class balancing, model tuning, a single final evaluation on the test set the preprocessing experiments and a comparison of validation methods. Every decision and its numbers are listed in `docs/report_requirements.md`. The IEEE report and the presentation are still to be written.


## Dataset

- Dataset: **wine_quality** (OpenML, ID 287)
- Link: [OpenML - Wine Quality Dataset](https://www.openml.org/d/287)
- 6,497 observations, 11 input features, and the target variable `quality`.
- Reference: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis.
    *Modeling wine preferences by data mining from physicochemical properties*.


## Repository Structure

```text
wine-quality-classification/
├── .github/
│   └── workflows/
│      └── quality.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── presentation/
├── reports/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── steps/
│   │   ├── s1_inspect_data.py
│   │   ├── s2_exploratory_data_analysis.py
│   │   ├── s3_preprocessing.py
│   │   ├── s4_discriminant_analysis.py
│   │   ├── s5_pca.py
│   │   ├── s6_class_balancing.py
│   │   ├── s7_model_tuning.py
│   │   ├── s8_final_evaluation.py
│   │   ├── s9_preprocessing_experiments.py
│   │   └── s10_resampling_comparison.py
│   ├── config.py
│   ├── data_loading.py
│   ├── evaluation.py
│   ├── models.py
│   └── outputs.py
│
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
│
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements-dev.txt
└── requirements.txt
```


## Setup

```bash
# 1. Clone the repository:

    git clone <repository-url>
    cd wine-quality-classification

# 2. Create and activate the virtual environment (venv):

    # Linux and macOS
    python3 -m venv .venv
    source .venv/bin/activate

    # Windows
    python -m venv .venv
    .venv\Scripts\Activate.ps1

# 3. Install the project requirements:

    pip install -r requirements-dev.txt

# 4. Set up pre-commit hooks:

    pre-commit install
```

To manually run all configured checks:

```bash
pre-commit run --all-files
```


## How to run

The project is organized into sequential steps, each responsible for a stage of the workflow.

Run the commands below from the repository root, with the virtual environment activated.

```bash
python -m src.steps.s1_inspect_data
python -m src.steps.s2_exploratory_data_analysis
python -m src.steps.s3_preprocessing
python -m src.steps.s4_discriminant_analysis
python -m src.steps.s5_pca
python -m src.steps.s6_class_balancing
python -m src.steps.s7_model_tuning
python -m src.steps.s8_final_evaluation
python -m src.steps.s9_preprocessing_experiments
python -m src.steps.s10_resampling_comparison
```

## Working with Branches

All development must be done in separate branches. Direct commits to `main` are not allowed.

**1. Before starting a new task, create a branch**

```bash
# Update your local `main` branch:

    git switch main
    git pull origin main

# Create a new branch following the naming convention:

    git switch -c type/branch-description
```

**2. Commit and push your changes**

```bash
# After making your changes:

    git add .
    git commit -m "prefix: describe your changes"
    git push -u origin type/branch-description
```

**3. Open a Pull Request**

On GitHub, open a Pull Request from your branch into `main`. Wait for the automated quality checks and request a review from another contributor. Once the required approval is obtained and all checks pass, your changes can be merged into `main`.

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.


## Code Quality and Formatting

This project uses **Black** for code formatting and **Ruff** for linting.

Both tools are integrated through **pre-commit**, which automatically runs them before each Git commit. Ruff applies available automatic fixes, while Black ensures consistent code formatting.

No editor-specific configuration is required. The checks are managed at the repository level.


## Members

- João Braga
- Lucas Mendes
- Mateus
- Ricardo Senna
- Samuel Rozas