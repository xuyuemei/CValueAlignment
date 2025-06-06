import jsonlines
from get import *
from deal import check

file_ok=jsonlines.open("/home/qzh/Value_cn/dataset/dataset_detail/deepseek2000_cr.jsonl_fail1.jsonl","a")
# file_fail=jsonlines.open("/home/qzh/Value_cn/dataset/dataset_detail/deepseek2000_cr_fail.jsonl","a")
# /home/qzh/Value_cn/dataset/dataset_detail/gpt2000.jsonl
index=0
words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
words_to_keep_cn=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]

with open("/home/qzh/Value_cn/dataset/dataset_detail/merge.jsonl_fail.jsonl", "r+", encoding="utf8") as f:

    for item in jsonlines.Reader(f):
        index+=1
        # if index<=1522:
        #     continue
        print(index)
        # if index>298 and index<1070:
        #     continue
        text=item["text"]
        tran_dict={}
        for key in words_to_keep:
            tran_dict[key]=item[words_to_keep_cn[words_to_keep.index(key)]]
        # print()
        new=get_answer(en_correct(item["text"],tran_dict))
        new["text"]=text
        jsonlines.Writer.write(file_ok,new)
        
        # print(index,new)
        # result=check(new,"en")


        # if result[1]==True:
        #     item.update(result[0])
        #     print("true",item)
        #     jsonlines.Writer.write(file_ok,item) 
        # else:
        #     item.update(result[0])
        #     print("false",item)
        #     jsonlines.Writer.write(file_fail,item) 
