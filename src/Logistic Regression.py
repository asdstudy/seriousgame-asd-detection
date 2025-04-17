from sklearn.model_selection import GridSearchCV, train_test_split, KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_curve, roc_auc_score, auc, make_scorer, precision_score, 
    recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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

# Configuração dos hiperparâmetros para o GridSearch
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Controle de regularização
    'penalty': ['l2', 'none'],  # Tipo de penalização (L2 ou sem penalização)
    'solver': ['lbfgs', 'saga'],  # Solvers compatíveis
    'class_weight': [None, 'balanced']  # Peso das classes
}

# Instanciar o modelo
logistic_model = LogisticRegression(random_state=42, max_iter=1000)

# Configurar o GridSearchCV
grid_search_rl = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

# Ajustar o modelo aos dados
grid_search_rl.fit(X_val_scaler, y_val)

# Exibir os melhores hiperparâmetros
print(f"Melhores hiperparâmetros encontrados: {grid_search_rl.best_params_}")

best_params_rl = grid_search_rl.best_params_

# Configurar o modelo com os melhores hiperparâmetros
best_rl = LogisticRegression(
    C=best_params_rl['C'],
    penalty=best_params_rl['penalty'],
    solver=best_params_rl['solver'],
    class_weight=best_params_rl['class_weight'],
    random_state=42,
    max_iter=1000
)

best_rl.fit(X_train_scaler, y_train)

y_pred = best_rl.predict(X_test_scaler)

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

y_prob_knn = best_rl.predict_proba(X_test_scaler)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'ROC - KNN (AUC = {roc_auc_knn:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC curve - Regressão logística')
plt.legend(loc='lower right')
plt.show()

plot_learning_curve(best_rl, X_train_scaler, y_train, title="Learning Curve for Deep Learning")

#Regressão Logística - K-fold

# Melhores hiperparâmetros do GridSearch
best_params_rl = grid_search_rl.best_params_

# Configurar o modelo com os melhores hiperparâmetros
logistic_model = LogisticRegression(
    C=best_params_rl['C'],
    penalty=best_params_rl['penalty'],
    solver=best_params_rl['solver'],
    class_weight=best_params_rl['class_weight'],
    random_state=42,
    max_iter=1000
)

# Configuração do k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

results = cross_validate(logistic_model, X_scater, df_Y, cv=kf, scoring=scoring, n_jobs=-1) #Ajustar bases de dados

# Resultados médios e desvios padrões
print(f"Accuracy médio: {np.mean(results['test_accuracy']):.2f} ± {np.std(results['test_accuracy']):.2f}")
print(f"Precision médio: {np.mean(results['test_precision']):.2f} ± {np.std(results['test_precision']):.2f}")
print(f"Recall médio: {np.mean(results['test_recall']):.2f} ± {np.std(results['test_recall']):.2f}")
print(f"F1-Score médio: {np.mean(results['test_f1']):.2f} ± {np.std(results['test_f1']):.2f}")