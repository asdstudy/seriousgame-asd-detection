from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from learning_curve import plot_learning_curve



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


param_grid_mlp = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh', 'logistic'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

mlp = MLPClassifier(max_iter=500, random_state=42)

grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Ajustar o modelo ao dataset
grid_search_mlp.fit(X_val_scaler, y_val)

print("Melhores hiperparâmetros encontrados:")
print(grid_search_mlp.best_params_)

best_params_mlp = grid_search_mlp.best_params_

best_mlp = MLPClassifier(max_iter=500, random_state=42,
    hidden_layer_sizes=best_params_mlp['hidden_layer_sizes'],
    activation=best_params_mlp['activation'],
    solver=best_params_mlp['solver'],
    alpha=best_params_mlp['alpha'],
    learning_rate=best_params_mlp['learning_rate'])

best_mlp.fit(X_train_scaler, y_train)
y_pred = best_mlp.predict(X_test_scaler)

acuracia = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {acuracia:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(3, 2))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()

y_prob_mlp = best_mlp.predict_proba(X_test_scaler)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_mlp)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'ROC - KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve - MLP')
plt.legend(loc='lower right')
plt.show()

plot_learning_curve(best_mlp, X_train_scaler, y_train, title="Learning Curve for MLP Classifier")

best_params_mlp = grid_search_mlp.best_params_

best_mlp = MLPClassifier(max_iter=500, random_state=42,
    hidden_layer_sizes=best_params_mlp['hidden_layer_sizes'],
    activation=best_params_mlp['activation'],
    solver=best_params_mlp['solver'],
    alpha=best_params_mlp['alpha'],
    learning_rate=best_params_mlp['learning_rate'])

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

results = cross_validate(best_mlp, X_scater, df_Y, cv=kf, scoring=scoring, n_jobs=-1) #Ajustar bases de dados

# Resultados médios e desvios padrões
print(f"Accuracy médio: {np.mean(results['test_accuracy']):.2f} ± {np.std(results['test_accuracy']):.2f}")
print(f"Precision médio: {np.mean(results['test_precision']):.2f} ± {np.std(results['test_precision']):.2f}")
print(f"Recall médio: {np.mean(results['test_recall']):.2f} ± {np.std(results['test_recall']):.2f}")
print(f"F1-Score médio: {np.mean(results['test_f1']):.2f} ± {np.std(results['test_f1']):.2f}")