import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

data = load_breast_cancer()
data_df = pd.DataFrame(data = data.data,columns = data.feature_names)
#print(data_df.head())
x = data.data
y = data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(solver='liblinear')

'''
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
model.fit(X_scaled, y)
'''

model.fit(x_train, y_train)
y_pred = model.predict(x_test)
#print(y_pred)
#Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#F1_metrics
f1 = f1_score(y_test, y_pred)
print("F1_score:",f1)

# auc scores
auc_score = roc_auc_score(y_test, y_pred)
print("Area Under The Curve:", auc_score)

#Formula: Precision = TP / (TP + FP)
precision= precision_score(y_test, y_pred)
print("Precicion:", precision)

#Formula: Specificity = TN / (TN + FP)
specificity = recall_score(y_test, y_pred, pos_label=0)
print("Specificity:", specificity)

#Formula: Sensitivity = TP / (TP + FN)
sensitivity = recall_score(y_test, y_pred)
print("Sensitivity:", sensitivity)
