# ⚽ FIFA World Cup Match Outcome Predictor

A Random Forest classifier that predicts World Cup match results — Home Win, Away Win, or Draw — using 50 years of historical data, with per-match win probabilities for any team matchup.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter)](https://jupyter.org)

---

## Dataset

Historical FIFA World Cup match data spanning ~50 years (1974–2022), covering all group stage and knockout matches across 12 tournaments.

---

## Features Engineered

| Feature | Description |
|---|---|
| `home/away_winrate` | Historical win rate per team |
| `home/away_avg_goal_diff` | Average goal difference (attack vs defence proxy) |
| `h2h_winrate` | Head-to-head record between the two teams |
| `stage_weight` | Match importance: Group Stage (1) → Final (6) |
| `year` | Accounts for team evolution across eras |

> All stats computed from matches *before* the current one — no data leakage.

---

## Model

```
Algorithm  : Random Forest
Train/Test : 80% / 20%
```

---

## Results

**Overall Accuracy: 38%**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Away Win | 0.43 | 0.34 | 0.38 | 56 |
| Draw | 0.35 | 0.33 | 0.34 | 52 |
| Home Win | 0.36 | 0.46 | 0.40 | 59 |
| **Weighted Avg** | **0.38** | **0.38** | **0.38** | **167** |

> Football is one of the most unpredictable sports in the world — low scoring, frequent upsets, and high variance make it notoriously hard to model. Professional sportsbooks with massive data pipelines and full-time quants still mispredict roughly half of all matches. A 38% accuracy on a 3-class problem (random baseline: 33%) reflects that reality honestly, not a flaw in the model.

**Feature Importance**

| Feature | Importance |
|---|---|
| `year` | 0.1618 |
| `away_winrate` | 0.1300 |
| `home_avg_goal_diff` | 0.1222 |
| `home_winrate` | 0.1177 |
| `away_team_enc` | 0.1148 |
| `home_team_enc` | 0.1137 |
| `away_avg_goal_diff` | 0.1125 |
| `h2h_winrate` | 0.0785 |
| `stage_weight` | 0.0487 |

> Features are fairly evenly distributed — no single feature dominates, which explains why the model struggles to find a strong signal. `year` being the top feature suggests team strength shifts significantly across decades.

---

## Prediction Example

```python
predict_match("Brazil", "Germany", stage="Final")

# Brazil vs Germany | Final 2026
# Predicted : Home Win
#   Home win :  59.5%
#   Away win :  22.4%
#   Draw     :  18.1%
```
