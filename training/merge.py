import pandas as pd
import numpy as np
import math

# 读取CSV文件
file_path = "/home/qzh/Value_cn/dataset/llmlabel.csv"
with open("/home/qzh/Value_cn/latest/dataset/200/deepseek200.csv", "r", encoding="utf-8") as file:
    labels1 = pd.read_csv(file) 
    labels1 = labels1.iloc[:, 1:].fillna(0).to_numpy() # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/200/deepseek_cr_200.csv", "r", encoding="utf-8") as file:
    labels2 = pd.read_csv(file)  
    labels2 = labels2.iloc[:, 1:].fillna(0).to_numpy()  # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/200/gpt4_cr_200.csv", "r", encoding="utf-8") as file:
    labels3 = pd.read_csv(file)  
    labels3 = labels3.iloc[:, 1:].fillna(0).to_numpy()   # 去掉前两列

with open("/home/qzh/Value_cn/latest/dataset/200/gpt4_200.csv", "r", encoding="utf-8") as file:
    alldata = pd.read_csv(file)  
    labels4 = alldata.iloc[:, 1:].fillna(0).to_numpy()   # 去掉前两列


a=np.full((4, 12), 0.25)
results=np.full((200, 12), 0)    

for j in range(0, 200):
    label1=labels1[j]
    label2=labels2[j]
    label3=labels3[j]
    label4=labels4[j]

    for i in range(0, 12):
        votes = sum([label1[i], label2[i], label3[i], label4[i]])
        if votes>2:
            results[j][i]=1
        elif votes<2:
            results[j][i]=0
        else:
            results[j][i]=2
            # lst = [-1.0 if x == 0.0 else x for x in [label1[i], label2[i], label3[i], label4[i]]]
            # print(lst*betas)
            # flag+=1
            # result = sum([a * b for a, b in zip(lst, betas)])
            # result = 0 if result <= 0 else 1
# results=results.T      
# labels1=labels1.T      

# print(results.shape,labels1.shape)

for (i,label) in enumerate([labels1,labels2,labels3,labels4]):
    for k in range(0,12):
        equal_counts = np.sum(results[:, k] == label[:, k])
        count = np.sum(results[:, k] == 0)+np.sum(results[:, k] == 1)
        a[i,k]=equal_counts/count
    # break
    # 
print(a.T)

#final_annotation
# from scipy import stats

# z_scores = stats.zscore(a.T, axis=0)  # axis=0 表示按列
texts=alldata.iloc[:, 0].to_list()

results=[]
# print(z_scores)
for j in range(0, 200):
    label1=labels1[j]
    label2=labels2[j]
    label3=labels3[j]
    label4=labels4[j]

    result=[texts[j]]

    for i in range(0, 12):
        betas=a.T[i]
        
        
        votes = sum([label1[i], label2[i], label3[i], label4[i]])
        if votes>2:
            result.append(1)
        elif votes<2:
            result.append(0)
        else:
            lst = [-1.0 if x == 0.0 else x for x in [label1[i], label2[i], label3[i], label4[i]]]
            print("lst",lst,betas)
            arr=[100*a * b for a, b in zip(lst, betas)]
            tmp=math.fsum(arr)
            # print("A",A)
            # print(results[j][i])
            tmp=0 if tmp <= 0 else 1
            result.append(tmp)
    results.append(result)
print(results)
import csv
header=["text","Prosperity","Democracy","Civility","Harmony","Freedom","Equality","Justice","Rule of Law","Patriotism","Dedication","Integrity","Friendliness"]
with open("/home/qzh/Value_cn/latest/dataset/200/llmlabeled200.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(results)
    
    # 写入所有文本数据
# print("CSV 文件已生成！")
