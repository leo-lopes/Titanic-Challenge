#  Spaceship Titanic — Kaggle Challenge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leo-lopes/Titanic-Challenge/blob/main/Titanic.ipynb)
![Python](https://img.shields.io/badge/Python-3.7-blue)
![Score](https://img.shields.io/badge/Kaggle%20Score-79%25-green)
![Leaderboard](https://img.shields.io/badge/Leaderboard-86%25-orange)

## Overview

This notebook tackles the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) competition on Kaggle — a binary classification problem where the goal is to predict which passengers were transported to an alternate dimension after the spaceship collided with a spacetime anomaly.

**Final score: 79% accuracy** (leaderboard top: 86%)

---

## The Problem

In the year 2912, the Spaceship Titanic carried nearly 13,000 passengers on its maiden voyage. A collision with a hidden spacetime anomaly caused almost half of the passengers to be transported to another dimension. Given passenger records recovered from the ship's damaged computer system, the task is to predict which passengers were transported.

---

## Dataset

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 8,693 | Training set with ground truth (`Transported`) |
| `test.csv` | 4,277 | Test set for prediction |

### Features

| Feature | Type | Description |
|---------|------|-------------|
| `PassengerId` | string | Unique identifier `gggg_pp` (group + position) |
| `HomePlanet` | categorical | Planet of departure (Europa, Earth, Mars) |
| `CryoSleep` | boolean | Whether passenger was in suspended animation |
| `Cabin` | string | Cabin number `deck/num/side` |
| `Destination` | categorical | Destination planet |
| `Age` | float | Passenger age |
| `VIP` | boolean | Whether the passenger paid for VIP service |
| `RoomService` | float | Amount billed at Room Service |
| `FoodCourt` | float | Amount billed at Food Court |
| `ShoppingMall` | float | Amount billed at Shopping Mall |
| `Spa` | float | Amount billed at Spa |
| `VRDeck` | float | Amount billed at VR Deck |
| `Name` | string | Passenger name |
| `Transported` | boolean | **Target** — whether the passenger was transported |

---

## Notebook Structure

### 1. Data Analysis & Visualization
- Exploratory Data Analysis (EDA) with Pandas
- Distribution plots and correlation heatmaps using Matplotlib, Seaborn and Plotly
- Key insights from feature relationships

### 2. Prediction Models
Three approaches were tested and compared:

| Model | Accuracy |
|-------|----------|
| Neural Network (best) | **79.8%** |
| Support Vector Machine | 79.2% |
| Random Forest | 76.8% |

### 3. Conclusions
- CryoSleep and spending habits (Spa, VRDeck, RoomService) were strong predictors of transportation
- Neural networks offered a marginal improvement over SVM but significantly outperformed Random Forest
- The gap to the leaderboard (86%) suggests feature engineering and hyperparameter tuning as main opportunities for improvement

---

## Tech Stack

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical computing
import matplotlib.pyplot as plt  # Static visualizations
import seaborn as sns        # Statistical visualizations
import plotly.express as px  # Interactive visualizations
```

---

## How to Run

### Option 1 — Google Colab (recommended)
Click the badge at the top of this README.

### Option 2 — Local
```bash
# Clone the repo
git clone https://github.com/leo-lopes/Titanic-Challenge.git
cd Titanic-Challenge

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly scikit-learn

# Download the data from Kaggle
# kaggle competitions download -c spaceship-titanic

# Open the notebook
jupyter notebook Titanic.ipynb
```

---

## Results & Next Steps

The 79% score is a solid baseline. Potential improvements to close the gap to the leaderboard:

- **Feature engineering** — extract `Deck`, `Side` and group information from `Cabin` and `PassengerId`
- **Ensemble methods** — combine predictions from multiple models
- **Hyperparameter tuning** — Grid Search / Bayesian optimization on the Neural Network
- **Missing value strategies** — more sophisticated imputation beyond mean/mode

---

## Competition Link

[Spaceship Titanic on Kaggle](https://www.kaggle.com/competitions/spaceship-titanic)
