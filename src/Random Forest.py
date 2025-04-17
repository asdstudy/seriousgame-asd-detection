from learning_curve import plot_learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score, auc
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns

df_X = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\df_X_2.json", lines=True)
X_val = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\X_val_2.json", lines=True)
X_test = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\X_test_2.json", lines=True)
X_train = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\X_train_2.json", lines=True)

df_Y = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\df_Y_2.json", lines=True)
y_val = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\y_val_2.json", lines=True)
y_test = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\y_test_2.json", lines=True)
y_train = pd.read_json("C:\\Users\\lct\\OneDrive\\Documentos\\GitHub\ESWA\\data\\y_train_2.json", lines=True) 

scaler = StandardScaler()
X_scater = scaler.fit_transform(df_X)
X_val_scaler = scaler.transform(X_val)
X_test_scaler = scaler.transform(X_test)
X_train_scaler = scaler.fit_transform(X_train)


param_grid = {
    'n_estimators': [50, 100, 200],        # Número de árvores
    'max_depth': [None, 10, 20, 30],      # Profundidade máxima
    'min_samples_split': [2, 5, 10],      # Número mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],        # Número mínimo de amostras em uma folha
    'bootstrap': [True, False]            # Método de amostragem (com ou sem reposição)
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Ajustar o modelo ao dataset
grid_search.fit(X_val_scaler, y_val)

print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

best_params_rf = grid_search.best_params_
best_rf = RandomForestClassifier(
    n_estimators=best_params_rf['n_estimators'],
    max_depth=best_params_rf['max_depth'],
    min_samples_split=best_params_rf['min_samples_split'],
    min_samples_leaf=best_params_rf['min_samples_leaf'],
    bootstrap=best_params_rf['bootstrap'],
    random_state=42
)

best_rf.fit(X_train_scaler, y_train)
importancias_features = best_rf.feature_importances_
importancias_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importancias_features})
importancias_df = importancias_df.sort_values(by='Importance', ascending=False)
y_pred = best_rf.predict(X_test_scaler)
acuracia = accuracy_score(y_test, y_pred)
precision_score
recall_score
make_scorer
print(f'Model Accuracy: {acuracia:.2f}')
print(importancias_df)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3, 2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

y_prob_rf = best_rf.predict_proba(X_test_scaler)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_rf)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'ROC - KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

plot_learning_curve(best_rf, X_train_scaler, y_train, title="Learning Curve for RF Classifier")

best_params = grid_search.best_params_

best_rf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    bootstrap=best_params['bootstrap'],
    random_state=42
)

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

results = cross_validate(best_rf, X_scater, df_Y, cv=kf, scoring=scoring, n_jobs=-1)

# Resultados médios e desvios padrões
print(f"Accuracy médio: {np.mean(results['test_accuracy']):.2f} ± {np.std(results['test_accuracy']):.2f}")
print(f"Precision médio: {np.mean(results['test_precision']):.2f} ± {np.std(results['test_precision']):.2f}")
print(f"Recall médio: {np.mean(results['test_recall']):.2f} ± {np.std(results['test_recall']):.2f}")
print(f"F1-Score médio: {np.mean(results['test_f1']):.2f} ± {np.std(results['test_f1']):.2f}")