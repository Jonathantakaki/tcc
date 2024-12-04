# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 19:14:38 2024

@author: Usuario
"""


import pandas as pd # manipulação de dados em formato de dataframe



populacao = pd.read_csv('pop_mundial.csv', delimiter=',')


df_pivot = populacao.pivot_table(index=['Country Name', 'Country Code'], 
                                 columns='Series Name', 
                                 values='2023 [YR2023]', 
                                 aggfunc='sum')  # Escolha 'sum' para somar duplicatas ou 'mean' para média

# Renomear as colunas
df_pivot = df_pivot.rename(columns={
    'Population, total': 'Population Total',
    'Population, female': 'Population Female',
    'Population, male': 'Population Male'
}).reset_index()

# Exibir o resultado final
print(df_pivot)                 