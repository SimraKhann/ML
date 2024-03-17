import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

data = pd.read_csv("/home/ibab/SEM_4/project/data/gvcf/final/server_final_updated_data_with_target_all_genes.tsv", delimiter="\t", usecols=lambda col: col != 0)
print(data)
data.drop("SampleName", axis=1, inplace=True)
print(data)
data.fillna(0, inplace=True)
#print(test_data.columns) 
x = data.drop("target",axis=1)
y = data["target"] 
kf = KFold(n_splits=10, shuffle=True, random_state=42)

accuracy_scores = []
roc_auc_scores = []
specificity_scores = []
sensitivity_scores = []
precision_scores = []
f1_scores = []

# Iterate through folds
for train_index, test_index in kf.split(x):
    X_train, X_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Create and train model
    model = LogisticRegression(solver='liblinear',C=0.1,penalty='l1')
    #model = xgb.XGBClassifier()
    #model = SVC(kernel='linear', C=1.0)
    #model = RandomForestClassifier(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    roc_auc = roc_auc_score(y_test, y_pred)
    roc_auc_scores.append(roc_auc)
    recall = precision_score(y_test, y_pred,pos_label=0)
    specificity_scores.append(recall)
   
    recall_1 = recall_score(y_test, y_pred)
    sensitivity_scores.append(recall_1)
   
    precision = precision_score(y_test, y_pred)
    precision_scores.append(precision)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)


accuracy_sd = np.std(accuracy_scores)
roc_auc_sd = np.std(roc_auc_scores)
sensitivity_sd = np.std(sensitivity_scores)
specificity_sd = np.std(specificity_scores)
precision_sd = np.std(precision_scores)
f1_sd = np.std(f1_scores)
# Print mean scores and standard deviations across folds
print("Mean Accuracy:", np.mean(accuracy_scores), "SD:", accuracy_sd)
print("Mean ROC AUC:", np.mean(roc_auc_scores), "SD:", roc_auc_sd)
print("Mean Sensitivity:", np.mean(sensitivity_scores), "SD:", sensitivity_sd)
print("Mean Specificity:", np.mean(specificity_scores), "SD:", specificity_sd)
print("Mean Precision:", np.mean(precision_scores), "SD:", precision_sd)
print("Mean F1 score:", np.mean(f1_scores), "SD:", f1_sd)

