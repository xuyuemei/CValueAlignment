import pandas as pd

# # 读取 jsonl 文件
input_file = '/home/qzh/Value_cn/code/value(1).json'

# # 使用 pandas 读取 jsonl 文件
# df = pd.read_excel(input_file)
df = pd.read_json(input_file,orient="index")

# # 保存为 Excel 文件
# output_file = '/home/qzh/Value_cn/dataset/dataset_detail/humanlabel2000.csv'
df.to_csv("output_file.csv", index=True)

# print(f"文件已保存为 {output_file}")

import pandas as pd

# 读取两个 CSV 文件
# df1 = pd.read_csv("/home/qzh/Value_cn/latest/dataset/2000/deepseek2000.csv")
# df2 = pd.read_json("/home/qzh/Value_cn/code/final_dataset/glm2000_ok.jsonl",lines=True)
# print(len(df2),len(df2))
# df2 = df2.drop_duplicates(subset=['text'])
# missing_texts = set(df1['text']) - set(df2_unique['text'])

# print("在 df1 但不在 df2_unique 中的 text:", missing_texts)

# 以 df1 的 text 列顺序对 df2 进行排序
# df2_sorted = df1[['text']].merge(df2, on='text', how='left')
# print(len(df2_sorted),len(df2_sorted))
# # 保存排序后的 CSV
# df2_sorted.to_csv("/home/qzh/Value_cn/latest/dataset/2000/glm2000.csv", index=False)

# print("/home/qzh/Value_cn/dataset/dataset_detail/deepseek2000_cr.csv")
