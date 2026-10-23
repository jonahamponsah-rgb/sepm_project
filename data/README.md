# Source data

Raw source datasets are not redistributed in this package. Saved predictions and results support verification without source-data downloads. To retrain, place these files in this directory, preserving the original row order and headers:

| File | Source | Required layout and outcome |
|---|---|---|
| `cervical_cancer.csv` | [UCI Cervical Cancer Behavior Risk](https://doi.org/10.24432/C5402W) | CSV with header, 72 rows; outcome `ca_cervix`, 0/1; 19 predictors in the supplied experiment file |
| `breast-cancer.data` | [UCI Breast Cancer](https://doi.org/10.24432/C51P4M) | Headerless, 286 rows; first column outcome; recurrence-events = 1; nine categorical predictors; `?` missing |
| `pima-indians-diabetes.csv` | [Pima dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/data) | Headerless, 768 rows, eight predictors then binary outcome; do not use a headered CSV unchanged |
| `haberman.csv` | [UCI Haberman Survival](https://doi.org/10.24432/C5XK51) | Headerless, 306 rows, three predictors then outcome; original outcome 2 = positive, 1 = negative |

The Pima source publication is [Smith et al. (1988)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/). File hashes for the exact original local inputs are in `expected_sha256.json`. File line endings may change hashes; row order and values affect the assigned folds. A repository download with a different predictor set or row order is not the identical experimental input.

Preprocessing is fitted within the partition used for each model. Breast uses most-frequent imputation, ordinal encoding with unknown category -1 and standardization. Diabetes treats zeros in predictor columns 1–5 as missing, uses iterative linear-regression imputation (500 iterations) and standardization. Cervical and Haberman use standardization without imputation or encoding.
