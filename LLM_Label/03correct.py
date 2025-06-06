import jsonlines
from get import *

file_ok=jsonlines.open("/home/qzh/Value_cn/dataset/dataset_detail/gpt2000_cr.jsonl","a")
file_fail=jsonlines.open("/home/qzh/Value_cn/dataset/dataset_detail/gpt2000.jsonl","a")

index=0
words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
words_to_keep_cn=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]

with open("/home/qzh/Value_cn/code/label/dd/gpt1000_cn.jsonl", "r+", encoding="utf8") as f:

    for item in jsonlines.Reader(f):
        index+=1
        # print(list(item.keys()))

        is_partial=False
        new={}

        # if "answer" in list(item.keys()) or "dict" in list(item.keys()): 
        #     jsonlines.Writer.write(file_fail,item) 
        #     continue

        for key in words_to_keep_cn:
            # print(key)
            if key in list(item.keys()):
                is_partial=True
                new[key]=item[key]
            else:
                new[key]=0
        
        # new["text"]=item["text"]
        new.update(item)
        print(new)
        jsonlines.Writer.write(file_ok,new)   



            

        # if index>169:
        #     break
        # tran_dict={}
        # for key in words_to_keep_cn:
        #     tran_dict[key]=item[words_to_keep[words_to_keep_cn.index(key)]]

        # new=get_answer(cn_correct(item["text"],tran_dict))
        
        # new.update(item)
        # print(index,new)
        # jsonlines.Writer.write(file_ok,new)       
