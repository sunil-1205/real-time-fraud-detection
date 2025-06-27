import pandas as pd
from sklearn.model_selection import train_test_split # used to divide the data into traning set and testing set
from  imblearn.over_sampling import SMOTE # used to handle class imbalance
from sklearn.ensemble import RandomForestClassifier # used to train the model
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score # used to evaluate the model
import joblib # used to save the model
import xgboost as xgb # used to train the model

df = pd.read_csv('../data/processed_transactions.csv')

X = df.drop('Class', axis=1) # features
y = df['Class'] # target variable

X_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # split the data into training and testing sets

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)# handle class imbalance using SMOTE

print(f"Before resampled:{y_train.value_counts().to_dict()}")# prints the class distribution before resampling
print(f"After resampled:{pd.Series(y_train_resampled).value_counts().to_dict()}")# prints the class distribution after resampling

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled) # train the model
y_pred_rf = rf_model.predict(x_test)

print("Random Forest Classifier:")
print(confusion_matrix(y_test, y_pred_rf)) # prints confusion matrix
print(classification_report(y_test, y_pred_rf)) # prints classification report
print("ROC AUC Score:", roc_auc_score(y_test, rf_model.predict_proba(x_test)[:,1]))  # prints ROC AUC score


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_resampled, y_train_resampled) # train the model
y_pred_xgb = xgb_model.predict(x_test)

print("XGBoost Classifier:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))# prints classification report
print("ROC AUC Score:", roc_auc_score(y_test, xgb_model.predict_proba(x_test)[:,1]))  # prints ROC AUC score

joblib.dump(xgb_model, '../models/xgb_model.pkl')  # save the model
print("✅ Model saved to models/fraud_model.pkl")