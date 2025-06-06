import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
import pandas as pd

# # 读取 jsonl 文件
# input_file = '/home/qzh/Value_cn/dataset/deepseek_detail_plus.jsonl'

# # 使用 pandas 读取 jsonl 文件
# df = pd.read_json(input_file, lines=True)

# # 保存为 Excel 文件
# output_file = '/home/qzh/Value_cn/dataset/deepseek_detail_plus.csv'
# df.to_csv(output_file, index=False)

# print(f"文件已保存为 {output_file}")

pd.set_option('display.max_columns', None)  # 显示所有列
# pd.set_option('display.max_rows', None)     # 显示所有行

# data_xls = pd.read_excel('/home/qzh/Value_cn/latest/dataset/1000/llmrevise.xlsx')
# data_xls.to_csv('/home/qzh/Value_cn/latest/dataset/1000/llmlabeled1000_rj.csv', encoding='utf-8',index=None)
# print(data_xls)
# data_xls = pd.read_excel('/home/qzh/Value_cn/dataset/deepseek.xlsx')
# data_xls.to_csv('/home/qzh/Value_cn/dataset/deepseek.csv', encoding='utf-8')

golden_path="/home/qzh/Value_cn/latest/dataset/200/humanlabeled200.csv" 
llm_path="/home/qzh/Value_cn/latest/dataset/200/llmlabeledo200.csv" 
with open(golden_path, "r", encoding="utf-8") as file:
    golden_data = pd.read_csv(file) 
    golden_data = golden_data.iloc[:, 1:]  # 去掉前两列

with open(llm_path, "r", encoding="utf-8") as file:
    llm_data = pd.read_csv(file)  
    llm_data = llm_data.iloc[:, 1:]  # 去掉前两列

print(golden_data)
print("****************")
print(llm_data)
print("****************")

chunk=golden_data.fillna(0).to_numpy()
labels=chunk[:,:12]
print(labels)

# print(llm_data)
chunk=llm_data.fillna(0).to_numpy()
labels1=chunk[:,:12]
# labels2=chunk[:,12:24]
# labels3=chunk[:,24:36]
# labels4=chunk[:,36:]

# print(labels.shape,labels4.shape)

# print(f1_score(labels,labels4,average='samples'))
labels = labels.astype(int)
labels1 = labels1.astype(int)
# labels2 = labels2.astype(int)
# labels3 = labels3.astype(int)
# labels4 = labels4.astype(int)
print("labels 数据类型:", labels.dtype)
# print("labels1 数据类型:", labels4.dtype)
# print("labels 中的唯一值:", np.unique(labels))
# print("labels1 中的唯一值:", np.unique(labels4))
# print("labels 形状:", labels.shape)
# print("labels1 形状:", labels4.shape)

# print(f1_score(labels,labels1,average='samples', zero_division=1)) 
# print(f1_score(labels,labels2,average='samples', zero_division=1)) 
# print(f1_score(labels,labels3,average='samples', zero_division=1)) 
# print(f1_score(labels,labels4,average='samples', zero_division=1)) 

def cosine_similarity(mat_a, mat_b):
    similarities = []
    for row_a, row_b in zip(mat_a, mat_b):
        dot_product = np.dot(row_a, row_b)
        norm_a = np.linalg.norm(row_a)
        norm_b = np.linalg.norm(row_b)
        if norm_a == 0 or norm_b == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_a * norm_b)
        similarities.append(similarity)
    return np.mean(similarities)  # 返回所有行相似度的平均值

for labels1 in [labels1]:
    # 确保 labels 和 labels1 的数据类型正确
    labels = labels.astype(int)
    labels1 = labels1.astype(int)

    # accuracy_scores = [accuracy_score(labels[:, i], labels1[:, i]) for i in range(labels.shape[1])]


    # 计算逐列 F1 分数
    f1_scores = [f1_score(labels[:, i], labels1[:, i], average='binary') for i in range(labels.shape[1])]

    # # 计算平均 F1 分数
    avg_f1 = np.mean(f1_scores)

    print("逐列 f1 分数:", f1_scores)
    print("平均 f1 分数:", avg_f1)
