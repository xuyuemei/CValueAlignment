import pandas as pd

path="/home/qzh/Value_cn/dataset/humanlabel.csv"

df = pd.read_csv(path)
# print(df)
words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
words_to_keep_cn=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]

for (index,i) in enumerate(words_to_keep):
    # print(words_to_keep_cn[index]+'0',words_to_keep[index]+'0',words_to_keep_cn[index],words_to_keep[index])
    columns_data = df[[words_to_keep_cn[index],words_to_keep_cn[index]+'.1',words_to_keep_cn[index]+'.2']]
    columns_data['row_variance'] = 1-columns_data.var(axis=1)
    columns_data['text']=df['text']
    print(columns_data)
    columns_data.to_csv("/home/qzh/Value_cn/dataset/24humanlabels/"+words_to_keep[index]+".csv")

