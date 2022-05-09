import pandas as pd

RANDOM_SEED = 42

pd.set_option("display.max_rows", None, "display.max_columns", None)

df = pd.read_csv("./Datasets/Train_orig.csv", encoding='utf-8')[['2', '类别']]
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # Shuffle the DataFrame rows then reset indices

df['IsTypeOne'] = 0
df.loc[df['类别'] == '1,2', 'IsTypeOne'] = 1
df.loc[df['类别'] == '1, 2', 'IsTypeOne'] = 1
df.loc[df['类别'] == '1', 'IsTypeOne'] = 1

df['IsTypeTwo'] = 0
df.loc[df['类别'] == '1,2', 'IsTypeTwo'] = 1
df.loc[df['类别'] == '1, 2', 'IsTypeTwo'] = 1
df.loc[df['类别'] == '2', 'IsTypeTwo'] = 1

df_1 = df[['2', 'IsTypeOne']]
df_1 = df_1.rename(columns={"IsTypeOne": "IsType"})
df_1.to_csv('./Datasets/Train_orig_1.csv', index=False, quoting=2)

df_2 = df[['2', 'IsTypeTwo']]
df_2 = df_2.rename(columns={"IsTypeTwo": "IsType"})
df_2.to_csv('./Datasets/Train_orig_2.csv', index=False, quoting=2)
