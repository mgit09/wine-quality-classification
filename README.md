# Previsão da qualidade de vinho

Trabalho prático final da disciplina de Aprendizado de Máquina (2026/2), UFMG.

O projeto investiga a qualidade de vinhos (tinto e branco) a partir de 11 propriedades físico-químicas, aplicando análise exploratória, análise discriminante, reamostragem, PCA e comparação de modelos de aprendizado de máquina.

## Status

Em andamento...

## Dados

- Banco: **wine_quality** (OpenML, id 287)
- Link: [https://www.openml.org/d/287](https://www.openml.org/search?type=data&status=active&sort=nr_of_downloads&id=287)
- 6497 observações, 11 variáveis de entrada e a variável-alvo `quality`
- Referência: P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis.
  *Modeling wine preferences by data mining from physicochemical properties*.
  Decision Support Systems, 47(4):547-553, 2009.

## Estrutura do repositório

```text
analise-wine-quality/
├── apresentacao/
├── dados/
│   ├── brutos/
│   └── processados/
├── relatorio/
├── resultados/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Como executar

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

## Integrantes

- João Braga
- Lucas Mendes
- Mateus
- Ricardo Senna
- Samuel Rozas
