import pandas as pd

# 读取两个 CSV 文件
# df1 = pd.read_csv("/home/qzh/Value_cn/latest/dataset/200/humanlabeled200.csv")
# # df2 = pd.read_csv("/home/qzh/Value_cn/latest/dataset/200/gpt4_200.csv")
# df2 = pd.read_json("/home/qzh/Value_cn/code/final_dataset/gpt200_ok_!.jsonl", lines=True)

# # 以 df1 的 text 列顺序对 df2 进行排序
# df2_sorted = df1[['text']].merge(df2, on='text', how='left')

# # 保存排序后的 CSV
# df2_sorted.to_csv("/home/qzh/Value_cn/latest/dataset/200/deepseek_cr_200.csv", index=False)

# print("/home/qzh/Value_cn/latest/dataset/200/gpt4_200.csv")

data = pd.read_csv("/home/qzh/Value_cn/latest/dataset/2000/llmlabeled2000.csv", delimiter=',', encoding='utf-8',header=0)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data.replace(10, 1, inplace=True)
data=data.fillna(0)#random_state=42
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
input_data = data.iloc[int(len(data)-200):, :]

input_data.to_csv("/home/qzh/Value_cn/latest/dataset/200/llmlabeledo200.csv", index=False)