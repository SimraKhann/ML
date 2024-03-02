import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#Reading data
data = pd.read_csv("/home/ibab/SEM_4/project/data/gvcf/final/server_final_updated_data_with_target.tsv", delimiter="\t", usecols=lambda col: col != 0)

print(data)
data.drop("SampleName", axis=1, inplace=True)
data.fillna(0, inplace=True)
scaler = StandardScaler()

# Assuming the target variable is in the 'target' column
X = data.drop('target', axis=1)  # Features
y = data['target']                # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Define the parameter grid for hyperparameter tuning
param_grid = {
	  'n_estimators': [100, 200, 300],   # Experiment with different numbers of trees
    'max_depth': [None, 10, 20],        # Experiment with different maximum depths of trees
    'min_samples_split': [2, 5, 10],    # Experiment with different minimum samples required to split a node
    'max_features': ['auto', 'sqrt'],   # Experiment with different strategies for selecting the maximum number of features
    'criterion': ['gini', 'entropy'],   # Experiment with different impurity criteria
    'bootstrap': [True, False],         # Experiment with bootstrapping samples
      'ccp_alpha': [0.0, 0.1, 0.2]   

}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=15,  # 15-fold cross-validation
                           scoring='accuracy',
                           n_jobs=-1)  # Use all available CPU cores

# Fit the grid search to the data
grid_search.fit(X_train_scaled, y_train)

# Get the best estimator from grid search
best_rf_classifier = grid_search.best_estimator_

# Make predictions on the test set using the best estimator
y_pred = best_rf_classifier.predict(X_test_scaled)
# Calculate scores
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
sensitivity = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

#calculating SD
accuracy_sd = np.std(accuracy_scores)
roc_auc_sd = np.std(roc_auc_scores)
sensitivity_sd = np.std(sensitivity_scores)
precision_sd = np.std(precision_scores)
f1_sd = np.std(f1_scores)

# Print mean scores and standard deviations across folds
print("Mean Accuracy:", np.mean(accuracy_scores), "SD:", accuracy_sd)
print("Mean ROC AUC:", np.mean(roc_auc_scores), "SD:", roc_auc_sd)
print("Mean Sensitivity:", np.mean(sensitivity_scores), "SD:", sensitivity_sd)
print("Mean Precision:", np.mean(precision_scores), "SD:", precision_sd)
print("Mean F1 score:", np.mean(f1_scores), "SD:", f1_sd)

