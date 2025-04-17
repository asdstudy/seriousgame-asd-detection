#pip install pandas
#pip install numpy
#pip install scikit-learn
#pip install seaborn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

caminho_do_arquivo_json = 'C:\\Users\\lct\\OneDrive\\Área de Trabalho\\asd_data_collection_anonymized_with_id_2.json'
df = pd.read_json(caminho_do_arquivo_json)

mapeamento_genero = {'Male': 0, 'Female': 1}
mapeamento_dificuldade = {'Easy': 0, 'Medium': 1, 'Hard': 2}
mapeamento_jogo = {'Memory': 0, 'Painting': 1, 'Pairing with shadows': 2}
mapeamento_turma = {'G3': 0, 'G5': 1}
#mapeamento_profile = {'P1': 0, 'P2': 1, 'P3': 2}

df['gender'] = df['gender'].replace(mapeamento_genero)
df['game'] = df['game'].replace(mapeamento_jogo)
df['difficulty'] = df['difficulty'].replace(mapeamento_dificuldade)
df['class'] = df['class'].replace(mapeamento_turma)
#df['profile'] = df['profile'].replace(mapeamento_profile)

df = df.drop('child', axis=1)
df_Y = df[['asd']]
df_X = df.drop('asd', axis=1)

scaler = StandardScaler()
X_train_val, X_test, y_train_val, y_test = train_test_split(df_X, df_Y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.30, random_state=42)
df_X_scater = scaler.fit_transform(df_X)
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
X_val_scaler = scaler.transform(X_val)


def save_to_json(df, filename): 
    path = os.path.join('data', filename + '.json') 
    df.to_json(path, orient='records', lines=True)

save_to_json(df_X, "df_X_2")
save_to_json(df_Y, "df_Y_2")
save_to_json(X_train, "X_train_2")
save_to_json(y_train, "y_train_2")
save_to_json(X_val, "X_val_2")
save_to_json(y_val, "y_val_2")
save_to_json(X_test, "X_test_2")
save_to_json(y_test, "y_test_2")

#------------------------------------------- Correlação de Pearson ----------------------------------------------------------
df_X.corr()
plt.figure(figsize=(20, 16))
sns.set(font_scale=1.2)
ax = sns.heatmap(df_X.corr(), annot=True)
plt.show()

#----------------------------------------- PCA ----------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()
print("Cumulative Explained Variance:")
print(cumulative_explained_variance)

plt.plot(range(1, len(cumulative_explained_variance)+1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Main Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow Chart for PCA')
plt.show()