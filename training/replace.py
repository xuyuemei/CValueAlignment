import pandas as pd
import numpy as np

def cosine_similarity_mat(mat_a, mat_b):
    similarities = []
    for row_a, row_b in zip(mat_a, mat_b):
        print(row_a, row_b)
        dot_product = np.dot(row_a, row_b)
        norm_a = np.linalg.norm(row_a)
        norm_b = np.linalg.norm(row_b)
        if norm_a == 0 or norm_b == 0:
            similarity = 0
        else:
            similarity = dot_product / (norm_a * norm_b)
        similarities.append(similarity)
    return np.mean(similarities)  # 返回所有行相似度的平均值  

def cosine_similarity_vector(row_a, row_b):
    print(row_a, row_b)
    dot_product = np.dot(row_a, row_b)
    norm_a = np.linalg.norm(row_a)
    norm_b = np.linalg.norm(row_b)
    if norm_a == 0 or norm_b == 0:
        similarity = 0
    else:
        similarity = dot_product / (norm_a * norm_b)
    return similarity  # 返回所有行相似度的平均值

# 设置批处理大小
batch_size = 100  

# 读取CSV文件
file_path = "/home/qzh/Value_cn/dataset/llmlabel.csv"
with open("/home/qzh/Value_cn/latest/dataset/1000/deepseek1000.csv", "r", encoding="utf-8") as file:
    labels1 = pd.read_csv(file) 
    labels1 = labels1.iloc[:, 1:].to_numpy()  # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/1000/deepseek1000.csv", "r", encoding="utf-8") as file:
    labels2 = pd.read_csv(file)  
    labels2 = labels2.iloc[:, 1:].to_numpy()  # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/1000/gpt1000.csv", "r", encoding="utf-8") as file:
    labels3 = pd.read_csv(file)  
    labels3 = labels3.iloc[:, 1:].to_numpy()   # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/1000/gpt1000.csv", "r", encoding="utf-8") as file:
    alldata = pd.read_csv(file)  
    labels4 = alldata.iloc[:, 1:].to_numpy()   # 去掉前两列

# with open("/home/qzh/Value_cn/latest/dataset/1000/humanlabeled1000.csv", "r", encoding="utf-8") as file:
#     human_label = pd.read_csv(file)  
#     human_label = human_label.iloc[:, 1:].to_numpy()   # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/1000/llmlabeled1000.csv", "r", encoding="utf-8") as file:
    llm_slabel = pd.read_csv(file)  
    llm_slabel = llm_slabel.iloc[:, 1:].to_numpy()   # 去掉前两列

result_lists=[]
text=alldata.iloc[:, 0].to_list()

count2=0
count7=0
count75=0
entropy_all_all=[]
for j in range(0, 1000):
    # print("batch_size:",j)
    #四个labelers打的12维度的标
    label1=labels1[j]
    label2=labels2[j]
    label3=labels3[j]
    label4=labels4[j]

    # humanlabel=human_label[j]
    llmlabel=llm_slabel[j]
    # print()

    entropy_all=0
    for i in range(0,12):
        data=[label1[i],label2[i],label3[i]]        
        values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)

        # 计算熵
        entropy = np.sum(probabilities * np.log(probabilities))

        # 如果要标准信息熵，加负号
        entropy = -entropy  
        entropy_all+=entropy
        # print(data,entropy)

    if  entropy_all!=0:   
        entropy_all/=12
    else:
        entropy_all=0
        
    # print(entropy_all)
    entropy_all_all.append(entropy_all)
    # break


    result_list=[text[j]]
    if entropy_all>0.266:
        count7+=1
        result_list.extend([3,3,3,3,3,3,3,3,3,3,3,3])
    # elif entropy_all>=0.265 and count2<95:
    #     count2+=1
    #     # print(entropy_all)
    #     result_list.extend([3,3,3,3,3,3,3,3,3,3,3,3])
    else:
        llmlabel=[]
        
        for i in range(12):
            i_sum=label1[i]+label2[i]+label3[i]
            if i_sum>1:
               llmlabel.append(1)
            else:
                llmlabel.append(0)
        
        print(llmlabel)
        
        result_list.extend(llmlabel)
    # print(result_list)
    print(label1,label2,label3)
    # print(humanlabel)
    result_lists.append(result_list)
    # break

print(count2+count7)
counts, bin_edges = np.histogram(entropy_all_all, bins=5)

# 输出每个区间的数量分布
for i in range(len(bin_edges) - 1):
    print(f"区间 {bin_edges[i]:.2f} - {bin_edges[i+1]:.2f} ： {counts[i]} 个数")

    
    
        # break
import csv
with open("/home/qzh/Value_cn/latest/dataset/1000/llmlabeled1000_triple.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(result_lists)

# print("CSV 文件已生成！")
