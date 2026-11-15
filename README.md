# Wine Quality Prediction

Final project for the Machine Learning course (2026/2) at the Federal University of Minas Gerais (UFMG).

This project investigates wine quality (red and white wines) based on 11 physicochemical properties, applying exploratory data analysis, discriminant analysis, resampling techniques, Principal Component Analysis (PCA), and comparison of machine learning models.

## Status

In progress...

## Dataset

- Dataset: **wine_quality** (OpenML, ID 287)
- Link: [OpenML - Wine Quality Dataset](https://www.openml.org/d/287)
- 6,497 observations, 11 input features, and the target variable `quality`.
- Reference: P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis.  
  *Modeling wine preferences by data mining from physicochemical properties*.  
  Decision Support Systems, 47(4):547–553, 2009.

## Repository Structure

```text
wine-quality-classification/
├── data/
│   ├── raw/
│   └── processed/
├── presentation/
├── reports/
├── results/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run

```bash
git clone <url-do-repositorio>
cd analise-wine-quality

# Linux e macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Members

- João Braga
- Lucas Mendes
- Mateus
- Ricardo Senna
- Samuel Rozas
