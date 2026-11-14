# Feature Engineering and Selection (Titanic)

A Python script that turns raw Titanic passenger data into machine-learning features, selects the most useful ones with several statistical methods, and writes a report of the whole process.

## What it does
1. **Loads** the raw data (`data/titanic.csv`, downloaded automatically if missing)
2. **Cleans** missing values (Age, Embarked, Fare)
3. **Engineers features:** Title, FamilySize, IsAlone, HasCabin, Deck, TicketGroupSize, FarePerPerson, LogFare, IsChild, AgeXClass, NameLength
4. **Encodes categorical variables:** binary (Sex) and one-hot (Embarked, Title, Deck)
5. **Splits** 80/20 (stratified, `random_state=42`) and **scales** continuous features with `StandardScaler` fitted on the training set only
6. **Filters** near-constant features and highly correlated pairs
7. **Selects** features with six methods (ANOVA F-test, mutual information, chi-squared, RFE, random forest importance, Lasso). Features chosen by at least 3 methods are kept
8. **Compares** baseline, all-engineered and selected feature sets with Logistic Regression and Random Forest

## How to run
```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
python src/feature_engineering.py
```

In Google Colab:
```python
!git clone https://github.com/<your-username>/<repo-name>.git
%cd <repo-name>
!python src/feature_engineering.py
```

## Outputs (created by the run)
| File | Content |
|---|---|
| `docs/feature_engineering_report.md` | Report on the process and the selected features, with all numbers from the run |
| `docs/feature_scores.csv` | Votes and scores of every feature |
| `data/titanic_features.csv` | Selected features plus the target |
| `images/feature_votes.png` | Votes per feature |
| `images/selected_correlation.png` | Correlation between selected features |
| `images/model_comparison.png` | Test accuracy by feature set |

## Structure
```
README.md
requirements.txt
src/feature_engineering.py
data/       (raw data and selected-feature CSV)
docs/       (report and scores)
images/     (charts)
```

## Dataset
Kaggle *Titanic: Machine Learning from Disaster*, `train.csv` (891 passengers): https://www.kaggle.com/c/titanic/data
