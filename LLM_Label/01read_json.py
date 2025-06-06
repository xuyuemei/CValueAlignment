from openpyxl import Workbook
from openpyxl import load_workbook
import jsonlines
from get import *
from prompt import *

read_file="/home/qzh/Value_cn/dataset/dataset_detail/deepseek2000.jsonl"
file_ok=jsonlines.open("/home/qzh/Value_cn/code/final_dataset/glm2000.jsonl","a")

index=0
with open(read_file, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        index+=1
        if index<=1885:
            continue
        
        # print("第"+str(ok_inx)+"条：")
        text = item["text"]
        # print(text)
        result=get_answer(cn_prompt(text))
        if('answer' in result.keys() and result['answer']==None):
            continue
        result["text"]=text         
        print(result)
        jsonlines.Writer.write(file_ok,result)      

