import pandas as pd

df = pd.read_csv("./Datasets/Test_orig.tsv", sep='\t')[0:100]
df_2 = pd.read_csv("./Datasets/Test_orig.tsv", sep='\t')[100:200]

print(df + df_2)
