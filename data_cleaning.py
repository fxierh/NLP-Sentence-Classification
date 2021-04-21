import re
import time

import pandas as pd

pd.set_option("display.max_rows", 100, "display.max_columns", None, "max_colwidth", None)

df = pd.read_csv("./Datasets/Test_orig.tsv", sep='\t', encoding='utf-8')
# df = pd.read_csv("./Datasets/Test_processed.csv", encoding='utf-8')

t0 = time.time()
index_to_drop = []
for row in range(len(df) - 1):
    '''
    有问题的地方：
    数字.（换行）数个（包含零个）空格接数字
    vs.（换行）
    左括号接不带右括号的序列（换行）不带左括号的序列接右括号
    '''
    if re.search(r'\d\.$', df.loc[[row]].values[0][1]) and re.search(r'^[ ]*\d', df.loc[[row + 1]].values[0][1]) \
            or re.search(r'vs\.$', df.loc[[row]].values[0][1]) \
            or re.search(r'[({\[][^)}\]]*$', df.loc[[row]].values[0][1]) \
            and re.search(r'^[^({\[]*[)}\]]', df.loc[[row + 1]].values[0][1]):
        index_to_drop.append(row + 1)
print(index_to_drop)
print(len(index_to_drop))

for row in reversed(index_to_drop):
    df.at[row - 1, '2'] = df.loc[[row - 1]].values[0][1] + df.loc[[row]].values[0][1]
df.drop(index_to_drop, inplace=True)

df = df[df['2'] != '资料与']  # Drop all rows where the value in col "2" is "资料与"
df = df[df['2'] != '材料与']
df = df[df['2'] != '材料和']
df = df[df['2'] != ' 材料.']
df = df[df['2'] != ',']

df['2'] = df['2'].str.strip()  # Strip all the spaces of the column "2"

df.to_csv('./Datasets/Test_processed.csv', index=False, quoting=2)

dt = time.time() - t0
print(f'Time consumed: {dt / 60} min')
