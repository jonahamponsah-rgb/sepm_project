
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

#%%
cervical_cancer_classifiers = [
    {
        'name': 'Logistic Regression',
        'classifier': LogisticRegression(max_iter=10000, random_state=42, solver='liblinear'),
        'param_grid': {'C': [0.1, 1.0, 10]}
    },
    {
        'name': 'SVM',
        'classifier': SVC(random_state=42, probability=True),
        'param_grid': {'C': [0.01, 1, 2, 4], 'kernel': ['rbf', 'linear'], 'gamma':['auto', 'scale']}
    },

    {
        'name': 'XGBoost',
        'classifier': XGBClassifier(random_state=42, objective='binary:logistic'),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.3, 10], 'max_depth': [4, 6]}
    },
    {
        'name': 'GBM',
        'classifier': GradientBoostingClassifier(random_state=42),
        'param_grid': {'n_estimators': [100, 200, 300], 'learning_rate': [0.1, 0.3, 10], 'max_depth': [1, 2]}
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'max_depth': [2, 4]}
    },
]


breast_cancer_classifiers = [
    {
        'name': 'Logistic Regression',
        'classifier': LogisticRegression(max_iter=10000, random_state=42, solver='liblinear'),
        'param_grid': {'C': [0.04, 0.06, 2]}
    },
    {
        'name': 'SVM',
        'classifier': SVC(random_state=42, probability=True),
        'param_grid': {'C': [1, 10, 50], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': ['auto', 'scale']}
    },

    {
        'name': 'XGBoost',
        'classifier': XGBClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.3], 'max_depth': [1, 2]}
    },
    {
        'name': 'GBM',
        'classifier': GradientBoostingClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.3, 0.5], 'max_depth': [2, 4]}
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'max_depth': [None, 1, 2, 4]}
    },
]

diabetes_classifiers = [
    {
        'name': 'Logistic Regression',
        'classifier': LogisticRegression(max_iter=10000, random_state=42, solver='liblinear'),
        'param_grid': {'C':  [0.04, 0.06, 1, 2]}
    },
    {
        'name': 'SVM',
        'classifier': SVC(random_state=42, probability=True),
        'param_grid': {'C': [0.04, 0.06, 2], 'kernel': ['rbf', 'linear'], 'gamma':['auto', 'scale']}
    },
    {
        'name': 'XGBoost',
        'classifier': XGBClassifier(random_state=42, objective='binary:logistic'),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.2, 0.3], 'max_depth': [None, 4]}
    },
    {
        'name': 'GBM',
        'classifier': GradientBoostingClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'learning_rate': [0.1, 0.2, 0.3], 'max_depth': [1, 2]}
    },
    {
        'name': 'Random Forest',
        'classifier': RandomForestClassifier(random_state=42),
        'param_grid': {'n_estimators': [50, 100, 150], 'max_depth': [None, 4]}
    },
]


# Dataset-specific classifiers dictionary
classifiers_dict = {
    "Cervical Cancer": cervical_cancer_classifiers,
    "Breast Cancer": breast_cancer_classifiers,
    "Pima Diabetes": diabetes_classifiers,
 
}

