# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:14:30 2024

@author: Usuario
"""
!pip install kaggle
# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
import pingouin as pg # outro modo para obtenção de matrizes de correlações
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
from scipy.stats import zscore
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# %% atleta eventos

atleta_eventos = pd.read_csv('athlete_events.csv', delimiter=',')



# Selicionando apenas o jogos de verão da variavel season

atleta_eventos_sumer = atleta_eventos[atleta_eventos['Season'] != 'Winter']

# Primeiramente, vamos excluir as variáveis que não serão utilizadas

atleta_eventos_sumer = atleta_eventos_sumer.drop(columns=['ID', 'Games'])

# Calcular o IMC para todos os atletas (altura convertida para metros, se necessário)
atleta_eventos_sumer['Height_m'] = atleta_eventos_sumer['Height'] / 100
atleta_eventos_sumer['IMC'] = atleta_eventos_sumer['Weight'] / (atleta_eventos_sumer['Height_m'] ** 2)

#excluindo valores na das cariaveis height_m e weight
atleta_eventos_sumer = atleta_eventos_sumer.dropna(subset=['Height_m', 'Weight'])


# Estatísticas descritivas do IMC por sexo
imc_por_sport_sex = atleta_eventos_sumer.groupby(['Sport', 'Sex'])['IMC'].describe()



# %% population

population = pd.read_csv('population.csv', delimiter=',')

population_pivot = population.pivot_table(index=['Country Name', 'Country Code'], 
                                 columns='Series Name', 
                                 values='2023 [YR2023]', 
                                 aggfunc='sum')  # Escolha 'sum' para somar duplicatas ou 'mean' para média




population_pivot.reset_index(inplace=True)

population_pivot.rename(columns={
    'Country Name': 'pais',
    'Country Code': 'cod_pais',
    'Population, female': 'pop_feminina',
    'Population, male': 'pop_masculina',
    'Population, total': 'pop_total'
}, inplace=True)

print(population_pivot.columns)
# %% pib


pib = pd.read_csv('pib.csv', delimiter=',', on_bad_lines='skip')
pib['2023 [YR2023]'] = pib['2023 [YR2023]'].replace('..', np.nan)
pib = pib.dropna(subset='2023 [YR2023]').reset_index(drop=True) 
pib = pib.drop(columns=['Series Name', 'Series Code'])
pib.rename(columns={'2023 [YR2023]': 'Pib_2023_dolar '}, inplace=True)
pib = pib.iloc[:175].reset_index(drop=True) 
# %% gini

gini = pd.read_csv('economic_gini_index.csv', delimiter=',')


gini_pivot = gini.pivot_table(index=['Entity', 'Code'], 
                                 columns='Year', 
                                 values='gini', 
                                 aggfunc='sum')  # Escolha 'sum' para somar duplicatas ou 'mean' para média

# %% Visualizando informações sobre os dados e variáveis

print(atleta_eventos_sumer.info())

# %% Estatísticas univariadas

atleta_eventos_sumer.describe()

#%% Tratamento dos dados


 
# excluindo as variaveis nulas de age, height, weight
atleta_eventos_sumer = atleta_eventos_sumer.dropna(subset=['Age', 'Height', 'Weight'])

#substituindo valores nan por 0 
atleta_eventos_sumer['Medal'] = atleta_eventos_sumer['Medal'].fillna(0)


#Substituindo Valores na Coluna medal para numericos
atleta_eventos_sumer['Medal'] = atleta_eventos_sumer['Medal'].replace({'Gold': 3, 'Silver': 2, 'Bronze': 1})





#%% colunas numericas

colunas_numericas = atleta_eventos_sumer[['Age', 'Height', 'Weight', 'Year', 'Medal', 'IMC', 'Sex']]

print(colunas_numericas.info())

# Obtendo as estatísticas descritivas das variáveis

tab_descritivas = colunas_numericas.describe().T


#%% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore
colunas_numericas_pad = colunas_numericas.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(round(colunas_numericas_pad.mean(), 3))
print(round(colunas_numericas_pad.std(), 3))

#%% Gráfico 3D das observações

plt.figure(figsize=(10, 8))
sns.boxplot(x="IMC", y="Sport", data=atleta_eventos_sumer, palette="pastel", showfliers=False)
sns.swarmplot(x="IMC", y="Sport", data=atleta_eventos_sumer, color=".25")
plt.title("Distribuição e Dados Individuais do IMC por Esporte")
plt.xlabel("IMC")
plt.ylabel("Esporte")
plt.show()

#%%

sns.boxplot(atleta_eventos_sumer, x="IMC", y="Sport")
fig.show()


#%%

df = pd.read_csv('seu_arquivo.tsv', delimiter='\t')



