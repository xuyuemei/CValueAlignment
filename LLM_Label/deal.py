#把raw data转为处理后的数据
import re
import json
import ast
words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]


def deal_dict(input_string,lan):
    # print(input_string)
    if(lan=="en"):
        words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
    elif(lan=="cn"):
        words_to_keep=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]

    # 构建正则表达式，匹配指定的单词和数字
    words_regex = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_keep) + r')\b'
    number_regex = r'\b\d+\b'
    
    # 合并成一个整体的正则表达式
    full_regex = r'(' + words_regex + r'|' + number_regex + r')'
    
    # 使用正则表达式替换其他内容为空
    result = re.findall(full_regex, input_string)
    # print(result)
    
    data_dict = {}

    for i in words_to_keep:
        if i in result and result.index(i)+1!=len(result) and result[result.index(i)+1] in ["1" , "0"]:
            # print(result[result.index(i)+1] )
            data_dict[i]=int(result[result.index(i)+1])
        else:
            return None
    return data_dict

def check(dict,lan):
    text=dict["text"]
    if(lan=="en"):
        words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
    elif(lan=="cn"):
        words_to_keep=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]
    # print(dict)
    if "dict" in dict:
        temp_dict=deal_dict(dict["dict"],lan)
        if temp_dict is None:
            return [dict,False]
        else:
            dict=temp_dict
        
    ordered_dict={}

    # print(dict)
    for i in words_to_keep:
        if i in dict:
            if  "0" in str(dict[i]):
                ordered_dict[i]=0
            elif "1" in str(dict[i]):
                ordered_dict[i]=1
        else:
            return [dict,False]
    ordered_dict["text"]=text
    return [ordered_dict,True]



