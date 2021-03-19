import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("Results/Test_preds.csv")
df.drop(columns=['Probabilities'], inplace=True)

test_sentences_nb = len(df.index)

df = df[df['Predictions'] != df['True labels']]

print(f'Wrong predictions: {len(df.index)} out of {test_sentences_nb}')

df.to_csv("./Results/Test_wrong_predictions.csv", index=False)
