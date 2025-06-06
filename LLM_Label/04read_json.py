import jsonlines
from get import *
from deal import check

file_ok=jsonlines.open("/home/qzh/Value_cn/dataset/new_labeled/fail2find1.jsonl","a")
file_fail=jsonlines.open("/home/qzh/Value_cn/dataset/new_labeled/fail2find2.jsonl","a")

index=0

with open("/home/qzh/Value_cn/code/label/fall.jsonl", "r+", encoding="utf8") as f:

    for item in jsonlines.Reader(f):
        index+=1
        print(index)
        text=item["text"]
        new=get_answer(cn_prompt(text))  
        print(new) 
        new["text"]=text
        result=check(new,"cn")

        # item = {key + '0': value for key, value in item.items() if key!='text'}  
        # print(index,new)

        if result[1]==True:
            item.update(result[0])
            # print("true",item)
            jsonlines.Writer.write(file_ok,item) 
        else:
            item.update(result[0])
            # print("false",item)
            jsonlines.Writer.write(file_fail,item) 

        # jsonlines.Writer.write(file_ok,new)       
