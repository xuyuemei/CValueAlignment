import jsonlines
from deal import check

type="glm"
read_file="/home/qzh/Value_cn/code/final_dataset/gpt200_fail_3.jsonl"
write_ok=jsonlines.open("/home/qzh/Value_cn/code/final_dataset/gpt200_ok_!.jsonl","w")
write_fail=jsonlines.open("/home/qzh/Value_cn/code/final_dataset/gpt200_fail_!.jsonl","w")

with open(read_file, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        # print(check(item,"en"))
        result=check(item,"en")
        if result[1]==True:
            # continue
            jsonlines.Writer.write(write_ok,result[0]) 
        else:
            # print("s")
            # continue
            jsonlines.Writer.write(write_fail,result[0]) 
        # break
        # jsonlines.Writer.write(write_ok,item)       
