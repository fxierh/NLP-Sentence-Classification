import pandas as pd

df = pd.read_csv("./Datasets/Test_orig.tsv", sep='\t')

df['IsTypeOne'] = 0
df.loc[df['类别'] == 0, 'IsTypeOne'] = 1

df['IsTypeTwo'] = 0
df.loc[df['类别'] == 1, 'IsTypeTwo'] = 1

df_1 = df[['2', 'IsTypeOne']][0:20000]
df_1 = df_1.rename(columns={"IsTypeOne": "IsType"})
df_1.to_csv('./Datasets/Dataset_1.csv', index=False)

df_2 = df[['2', 'IsTypeTwo']][0:20000]
df_2 = df_2.rename(columns={"IsTypeTwo": "IsType"})
df_2.to_csv('./Datasets/Dataset_2.csv', index=False)
