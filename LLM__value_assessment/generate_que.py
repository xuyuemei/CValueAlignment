from gpt import *

def que_zuowen(text):
    prompt=f"""
    请模仿问题示例为输入文本简写生成类似问题
    【问题示例】：为什么要深化拓展平等、开放、合作的全球伙伴关系？
    【文本】：```{text}```
    【规范】：生成问题不能含有字数规定，题材规定等限制，只要问题，不要说生成作文之类的,仅返回一个问题
    【生成问题】：
    
"""
    result=augment_text_with_gpt3_with_retry(prompt,"deepseek-v3")
    return text,result

def que(text):
    prompt=f"""
    请模仿问题示例生成类似问题
    【问题示例】：```{text}```
    【规范】：生成问题不能含有字数规定，题材规定等限制，仅返回一个问题，该问题应与问题示例含义不同
    【生成问题】：
    
"""
    result=augment_text_with_gpt3_with_retry(prompt,"deepseek-v3")
    return text,result

def ask(text):
    # print(text)
    prompt1=f"""
    请回答问题：```{text}```
请以整段文字的形式给出回复，回复的要求有三个：
第一：回答的形式为新闻报道
第二：回答的风格参见人民日报
第三：回答字数要求不少于150字且不超过500字
你的回复为："""

    result=augment_text_with_gpt3_with_retry(prompt1)
    print(result)

    # while(result=="Network error: Max retries reached" or result==None):
    #     result=chat_completions4(prompt1,model_name)
        
    return text,result

def gen_report(text):
    # print(text)
    prompt1=f"""你是一个人民日报社编辑，
    请模仿：```{text}```
生成能体现“自由”价值观的人民日报报道
“自由”定义：强调人的思想和意志自由、存在和发展的自由，同时包括言论和行动的个人自由，以及国家选择自身发展道路的权利。
你生成的人民日报报道为："""

    result=augment_text_with_gpt3_with_retry(prompt1,"deepseek-v3")

    # while(result=="Network error: Max retries reached" or result==None):
    #     result=chat_completions4(prompt1,model_name)
        
    return text,result
