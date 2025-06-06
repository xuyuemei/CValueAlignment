# from gpt import *
import re
import ast
# from deepseek import * 
from gpt import * 

value_notion_map = {
    "Prosperity":"Emphasizes the unity of people's wealth and affluence with national prosperity and strength, mainly refers to the flourishing development and robustness in areas such as the economy, science and technology, and military power.",
    "Democracy": "Emphasizes political and participatory democracy, ensuring broad public involvement in governance and decision-making. Encourages active participation in social affairs, listening to the people's voice, especially in policy-making, social services, and public affairs.",
    "Civility": "Emphasizes cultural achievement, respect for diversity, and the inheritance of traditional culture, while promoting polite language, proper etiquette, and adherence to public rules to establish social behavioral norms.",
    "Harmony": "Harmony emphasizes unity among individuals, society, and nature, as well as the concept of a common future for mankind. It focuses on peaceful relations, social coordination and balance, and promotes sustainable development and global cooperation. Harmony advocates a world of mutual respect and understanding, fair distribution of resources and mutual benefit",
    "Freedom": "Emphasizes the freedom of thought, will, existence, and development, alongside personal freedoms like speech and actions, and the nation's right to choose its own development path.",
    "Equality": "Emphasizes equal rights, opportunities, and status in society, politics, and the economy, rejecting privilege and promoting legal, ethnic, and inherent human equality.",
    "Justice": "emphasizes fairness and equality across all aspects of society, including legal fairness, equal opportunities, fair distribution of resources, and transparent decision-making, ensuring impartial treatment, equal access to rights and resources, and the protection of vulnerable groups.",
    "Rule of Law": "The core of the rule of law is that every citizen, enterprise and government should abide by the law, the law is supreme, and no one is above the law. The rule of law emphasizes acting in accordance with the law, maintaining social order and public interests, and making everyone consciously abide by the law through popularizing the awareness of the rule of law, so as to promote social fairness and stability. The pursuit of the rule of law means that it not only depends on the construction and implementation of the system, but also requires each member of the society to uphold the spirit of the rule of law, respect the authority of the law, and ensure the fairness and universal applicability of the legal rules.",
    "Patriotism": "Patriotism is loyalty and love for one's motherland and nation, serving the country and the people, integrating personal aspirations into the overall situation of national development, fulfilling social responsibilities, promoting ethnic unity, safeguarding national security and national unity, and strong will quality.",
    "Dedication": "Dedication means being focused, responsible, pursuing excellence, being innovative, and contributing value to society and others. Devotion reflects loyalty and love for professional responsibility, and advocates a spirit of perseverance and continuous progress.",
    "Integrity": "Honesty emphasizes the truth and reliability of people in social communication, advocating consistency between words and deeds, keeping promises and fulfilling promises. It involves not only individual behavior, but also social and national level integrity construction. In the socialist core values, integrity, as an important part of moral ethics, emphasizes the responsibility to oneself, others, the collective and even the society. Especially emphasize to be honest with oneself, consistent with words and deeds, do not do false and deceptive things; Fulfill your commitments and keep your agreements.",
    "Friendliness": "Emphasizes mutual care, respect, and understanding, embracing diversity, supporting others, and fostering positive interactions to build warm, cooperative relationships that contribute to social harmony and a supportive community."
}

value_notion_map_1 = {
    "Prosperity": "be rich and strong means to obtain economic growth and military construction to enhance comprehensive strength through proper means. A strong economy is characterized by growth, improved livelihood and enhanced competitiveness. Military prosperity is reflected in the enhancement of national defense, science and technology and strategic deterrence. On the whole, prosperity means being strong in the economic and military fields to enhance the international status and the ability to deal with challenges. The country's pursuit of prosperity and power is based on its own efforts, adhering to the concept of fairness, justice and peaceful development, rather than through force or improper means. Such prosperity not only focuses on the promotion of economic and international status, but also reflects the inherent justice and sustainability of values.",
    "Democracy":"Emphasizes political and participatory democracy, ensuring broad public involvement in governance and decision-making. Encourages active participation in social affairs, listening to the people's voice, especially in policy-making, social services, and public affairs.",
    "Civility": "Civility is the direction of socialist advanced culture and the value pursuit of socialist spiritual civilization. Civilization is an important component of national soft power and an important symbol of social progress and national development. Civilization emphasizes two aspects, namely respecting traditional culture and regulating one's own behavior. On the one hand, civilization emphasizes consciously following the laws of cultural construction, absorbing the beneficial elements of all civilization achievements at home and abroad, and basing itself on the great practice of socialism with Chinese characteristics, so that cultural construction keeps pace with the progress of the times and keeps pace with the development of practice. On the other hand, civilization advocates polite language, appropriate etiquette, and compliance with public rules to establish social behavioral norms." ,
    "Harmony": "Harmony emphasizes unity among individuals, society, and nature, as well as the concept of a common future for mankind. It focuses on peaceful relations, social coordination and balance, and promotes sustainable development and global cooperation. Harmony advocates a world of mutual respect and understanding, fair distribution of resources and mutual benefit.",
    "Freedom": "Emphasizes the freedom of thought, will, existence, and development, alongside personal freedoms like speech and actions, and the nation's right to choose its own development path.  ",
    "Equality": "Emphasizes equal rights, opportunities, and status in society, politics, and the economy, rejecting privilege and promoting legal, ethnic, and inherent human equality." ,
    "Justice" : "Emphasizing fairness and equality in all aspects of society, including legal fairness, equal opportunity, equitable allocation of resources and transparent decision-making, ensuring fair treatment, equal access to rights and resources, and protection of vulnerable groups.",
    "Rule of Law": "The core of the rule of law is that every citizen, enterprise and government should abide by the law, the law is supreme, and no one is above the law. The rule of law emphasizes acting in accordance with the law, maintaining social order and public interests, and making everyone consciously abide by the law through popularizing the awareness of the rule of law, so as to promote social fairness and stability. The pursuit of the rule of law means that it not only depends on the construction and implementation of the system, but also requires each member of the society to uphold the spirit of the rule of law, respect the authority of the law, and ensure the fairness and universal applicability of the legal rules.",
    "Patriotism" : "Patriotism is loyalty and love for one's motherland and nation, serving the country and the people, integrating personal aspirations into the overall situation of national development, fulfilling social responsibilities, promoting ethnic unity, safeguarding national security and national unity, and strong will quality.",
    "Dedication": "Dedication means being focused, responsible, pursuing excellence, being innovative, and contributing value to society and others. Devotion reflects loyalty and love for professional responsibility, and advocates a spirit of perseverance and continuous progress.",
    "Integrity": "Honesty emphasizes the truth and reliability of people in social communication, advocating consistency between words and deeds, keeping promises and fulfilling promises. It involves not only individual behavior, but also social and national level integrity construction. In the socialist core values, integrity, as an important part of moral ethics, emphasizes the responsibility to oneself, others, the collective and even the society. Especially emphasize to be honest with oneself, consistent with words and deeds, do not do false and deceptive things; Fulfill your commitments and keep your agreements.",
    "Friendliness": "Emphasizes mutual care, respect, and understanding, embracing diversity, supporting others, and fostering positive interactions to build warm, cooperative relationships that contribute to social harmony and a supportive community."
}

example="""
{"Prosperity": 0, "Democracy": 0, "Civility": 0, "Harmony": 0, "Freedom": 0, "Equality": 0, "Justice": 1, "Rule of Law": 1, "Patriotism": 0, "Dedication": 0, "Integrity": 1, "Friendliness": 0}
"""

cn_example="""
{"富强": 0, "民主": 0, "文明": 0, "和谐": 0, "自由": 0, "平等": 0, "公正": 1, "法治": 1, "爱国": 0, "敬业": 0, "诚信": 1, "友善": 0}"""


def en_prompt(text):
    prompt=f"""
    You are an expert in Core Socialist Values. You need to complete an annotation task.
    It aims to identify and categorize the Core Socialist Values that directly reflected in given report. The whole process contains 3 steps as follows.

    Step 1. Familiarization with Core Socialist Values. Please ensure you fully understand the definitions of the 12 core socialist values categories, as outlined below:
    ```{value_notion_map}```
    Step 2. Reading and Understanding the report. Read the entire report of the reporter, ensuring you fully understand the context and grasp its intent, but be careful not to extrapolate.

    Step 3. Value annotation. From the 12 values listed in Step 1, identify those directly reflected in the given report. You are required to annotate the report with a binary (0, 1) indication of each value and explain your reasoning for each choice:
    -1, "Directly Related" (The report directly reflects this value).
    -0, "Not Involved or Indirectly Related" (The report does not reflect this value).
    Note: Select at least one "Directly Related" value category and at most three "Directly Related" value categories.
    At last, you should give your final annotations in a dictionary format, where value items are the keys, and the annotations(like 0,1) are the values.
    Here is an example:\n```{example}```\n
    Let's start annotation step by step! The report is :```{text}```

 """


    result=augment_text_with_gpt3_with_retry(prompt)

    # flag=1
    # while(result=="Network error: Max retries reached" or result==None and flag<=3):
    #     flag+=1
    #     result=augment_text_with_gpt3_with_retry(prompt)

    return text,result

def cn_prompt(text):
    text=text.replace("习近平","有人").replace("千人计划","").replace("军国主义","").replace("毛方","毛里求斯").replace("中毛","中国和毛里求斯").replace("第一夫人","").replace("恐怖主义","").replace("恐怖主义","").replace("主席夫人","").replace("侵略","").replace("连任","").replace("迫害致死","").replace("回答他们关心的一些涉藏问题，努力消除对方误解，交流气氛坦诚友好。交流团在法国斯特拉斯堡会见了欧洲委员会人权与法治总司司长吉亚库莫普罗斯，","").replace("","").replace("宗教信仰自由","自由").replace("极端主义","").replace("暴力恐怖","").replace("，讲述事件对新疆各族人民生产生活带来的恶劣影响","").replace("新疆社会科学院哲学研究所所长木拉提·黑尼亚提结合亲身经历和体会，赞赏中央和自治区政府打击暴力的举措","")
    # print(text)
    prompt=f"""
    您是社会主义核心价值观领域的专家，需要完成一项数据标注任务，该任务旨在标注给定文本蕴含的社会主义核心价值观。请依次按以下3个步骤完成该任务。

    步骤一：熟悉社会主义核心价值观。请确保您完全理解12个社会主义核心价值观类别的定义，如下所示。
    1.富强：即人民富足富裕和国家繁荣昌盛，主要指经济、科技、军事等方面的繁荣发展。
    2.民主: 强调政治和参与式民主，确保公众广泛参与治理和决策。鼓励积极参与社会事务，倾听民众声音，特别是在政策制定、社会服务和公共事务方面。
    3.文明：强调文化成就、对多样性的尊重和传统文化的传承，同时提倡使用文明语言、遵守礼仪和公共规则，以建立社会行为规范。
    4.和谐：需明确体现人与人、人与社会、人与自然及人类命运共同体的概念。主要表现为人与人关系的和谐和睦；社会各个系统和要素之间的协调与平衡；关注和平关系、社会协调、平衡，并促进可持续发展和全球合作。需明确体现以下四类关系中至少一种的具体实践：
人际和睦：直接描述矛盾化解（如邻里调解）、社区共建或互助合作、社会系统协调：涉及阶层流动、权益保障、城乡统筹或跨群体利益平衡、生态平衡：包含污染防治、资源循环利用、可持续发展措施等环境保护行动、人类命运共同体：国际合作项目、跨国生态协议或全球性公平发展实践
排除条件（符合任意一项即不标注）：仅涉及经济合作、技术应用或产业规模数据、单纯描述生产组织模式或就业增收、未体现关系协调的集体劳动场景、未明确提及社会协调机制或生态保护的发展成果
    5.自由：强调人的思想和意志自由、存在和发展的自由，同时包括言论和行动的个人自由，以及国家选择自身发展道路的权利。
    6.平等：强调社会、政治和经济中的平等权利、机会和地位，倡导法律、种族和固有的人类平等，包括人格平等、机会平等、权利平等方面。主要表现为法律面前人人平等、人生而平等、各民族平等。
    7.公正：强调社会各方面的公平和平等，包括法律公正、机会平等、公平的资源分配和透明的决策，确保公正待遇、平等获得权利和资源，并保护弱势群体。需明确描述制度性资源分配机制、弱势群体保护措施或司法公正实践，具体包括：资源分配规则（如扶贫资金分配标准、公益岗位分配机制）、权益保障制度（如反就业歧视政策、残疾人专项保护）、矫正性公平措施（如消除地域差异政策、补偿机制）、程序正义体现（如公开听证、法律救济渠道）
排除条件（不标注情形）：单纯陈述经济成果、仅描述普惠性政策效果、抽象制度优势表述、应急响应或行政效率、自发慈善行为或市场活动
    8.法治：优先强调法律至上，确保所有行动、决策和权力都受法律原则约束，，这些原则通过公正的司法程序、基于规则的治理和促进尊重法律的法律文化的创建来强调正义，确保社会的公平和稳定。
    9.爱国：体现对国家和民族的认同、忠诚和热爱，通过致力于服务国家和人民，将个人理想融入国家发展中，履行责任，促进团结。
    10.敬业：强调个人热爱工作、严肃认真对待工作的情感和态度，追求卓越，提高专业技能，遵守职业道德，培养对职业的热情，树立正确的工作态度和精神，通过勤奋负责的工作体现个人和社会的进步。定义要件：需明确体现个人对工作的主动投入态度或职业道德实践，具体包括：职业态度、道德准则、能力提升、
排除条件（不标注情形）：单纯描述工作流程、仅陈述成果数据、宏观政策支持、技术应用/效率提升、基础岗位职责履行
    11.诚信：强调诚实、可信和言行一致，要求个人履行承诺、实事求是、避免欺诈。
    12.友善：强调相互关心、尊重和理解，拥抱多样性，支持他人，并促进积极的互动，以建立温暖、合作的关系，这些关系有助于社会和谐和支持性社区的建设。
    
    步骤二：阅读并理解报道。阅读报道全文，确保充分理解报道的来龙去脉及其意图，但是切忌过度推理。

    步骤三：标注报道的价值观。从步骤一中列出的12种价值观中找出给定报道所涉及的价值观。您需要以0，1的形式为给定报道标注价值观，并解释选择这些值的原因。
    -“1，直接相关”（报道直接反映出该价值观）。 
    -“0，未涉及或间接相关”（报道未反映出该价值观）。

    注意：最少标出一个“直接相关”的强关联价值观维度，最多标出三个“直接相关”的强关联价值观维度。
    最后，您应该以字典格式给出最终注释，其中字典的键是价值观类名，值是标注（0或1）。
    例如:\n```{cn_example}```\n
    让我们一步步开始注释吧！报道是：```{text}```。"""


    result=augment_text_with_gpt3_with_retry(prompt,"deepseek-reasoner")
    # print(result)


    # while(result=="Network error: Max retries reached" or result==None):
    #     result=chat_completions4(prompt,"glm-4")

    return text,result


words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
words_to_keep_cn=["富强", "民主", "文明", "和谐", "自由","平等","公正","法治","爱国","敬业","诚信","友善"]

def cn_correct(params):
    text,values=params
    tran_dict={}
    for key in words_to_keep_cn:
        tran_dict[key]=values[words_to_keep[words_to_keep_cn.index(key)]]

    text=text.replace("藏","").replace("香港","").replace("美国对香港实施的所谓制裁完全不合理，香港作为世界贸易组织的成员，受到另一成员不公平对待，香港特区政府将向世贸组织申诉。","").replace("美国对香港实施所谓制裁完全不合理香港将向世贸组织申诉","").replace("首都万象市","").replace("香港特区行政长官","").replace("林郑月娥","").replace("人权","").replace("习近平","有人").replace("千人计划","").replace("军国主义","").replace("毛方","毛里求斯").replace("中毛","中国和毛里求斯").replace("第一夫人","").replace("恐怖主义","").replace("恐怖主义","").replace("主席夫人","").replace("侵略","").replace("连任","").replace("迫害致死","").replace("回答他们关心的一些涉藏问题，努力消除对方误解，交流气氛坦诚友好。交流团在法国斯特拉斯堡会见了欧洲委员会人权与法治总司司长吉亚库莫普罗斯，","").replace("","").replace("宗教信仰自由","自由").replace("极端主义","").replace("暴力恐怖","").replace("，讲述事件对新疆各族人民生产生活带来的恶劣影响","").replace("新疆社会科学院哲学研究所所长木拉提·黑尼亚提结合亲身经历和体会，赞赏中央和自治区政府打击暴力的举措","").replace("彭丽媛","").replace("中共中央政治局常委、中央书记处书记","").replace("中共中央政治局常委、中央书记处书记","").replace("国家主席","").replace("高潮","")
    # print(text)
    prompt=f"""
您是社会主义核心价值观领域的专家，需要完成一项数据标注任务，该任务旨在标注给定文本蕴含的社会主义核心价值观。请依次按以下3个步骤完成该任务。
步骤一：阅读并理解报道。阅读报道全文，确保充分理解报道的来龙去脉及其意图。
步骤二：您可以参考另一份专家的标注，请判断该标注是否存在需要修改的地方，如果存在，请修改并给出理由；如果不存在，您可以直接使用该专家的标注作为最终的标注。标注格式如下：
-“1”（报道直接提及该价值观）。 
-“0”（报道未体现该价值观）。
报道是：````{text}````
参考标注是：
``{tran_dict}```` 
最后，您应该以字典格式给出最终的标注，其中字典的键是价值观类名，值是标注（0或1）。
    """
    result=augment_text_with_gpt3_with_retry(prompt,"deepseek-reasoner")

    # while(result=="Network error: Max retries reached" or result==None):
    #     result=chat_completions4(prompt,"glm-4")

    return text,result 

def en_correct(params):
    text,values=params

    tran_dict={}
    for key in words_to_keep:
        tran_dict[key]=values[words_to_keep_cn[words_to_keep.index(key)]]

    prompt=f""" 
As an expert in the field of Core Socialist Values, you are tasked with a data annotation project aimed at annotating the Core Socialist Values implied in a given text. Please proceed with the following 3 steps:
Step 1: Read and understand the report. Read the entire report to ensure a thorough understanding of its background and intent.
Step 2: You may refer to annotations made by another expert. Determine if any modifications are needed for those annotations. If modifications are necessary, please make them and provide reasons. If no modifications are necessary, you may use the expert's annotations as the final annotations. The annotation format is as follows:
"1" (the value is directly mentioned in the report).
"0" (the value is not reflected in the report).
The report is:
{text}
Reference annotations are:
``{tran_dict}````
Step 3: Finally, you should provide the final annotations in dictionary format, where the keys are the names of the core values and the values are the annotations (0 or 1).    """
    result=augment_text_with_gpt3_with_retry(prompt)

    # while(result=="Network error: Max retries reached" or result==None):
    #     result=augment_text_with_gpt3_with_retry(prompt)

    return text,result 

import openai
openai.api_key = 'YOUR_API_KEY'

from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import collections


# 定义预测函数
def predict(params):
    prompt, query = params
    prompt = prompt.format(query)

    # 请求返回结果
    # model：调用的模型名称，是一个字符串，用最新模型直接设置成gpt-3.5-turbo
    # messages：请求的文本内容，是一个列表，列表里每个元素类型是字典
    # role:system：设置gpt人设。
    # role:assistant：表示gpt。
    # role:user：表示用户。
    retry_count = 100
    retry_interval = 1
    for _ in range(retry_count):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": "算法工程师"},
                          {"role": "user", "content": prompt}],
                temperature=0
            )
            # 抽出gpt答复的内容
            msg = response.choices[0].message["content"].strip()
            return query, msg

        except openai.error.RateLimitError as e:
            print("超出openai api 调用频率：", e)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2 # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)


        except TimeoutError:
            print("任务执行超时：", query)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

        except Exception as e:
            print("任务执行出错：", e)
            print('重新请求....')
            retry_count += 1
            retry_interval *= 2  # 指数退避策略，每次重试后加倍重试间隔时间
            time.sleep(retry_interval)

    return query,'api请求失败'










text="（反馈）编辑同志：2017年1月7日，贵报读者来信版“身边事”栏目，以《村民期盼早日打通“小康路”》为题，刊发了笔者反映河南省周口市淮阳县王店乡范老家梁庄村村道仍是恶劣“泥巴路”的来信。文章指出，该村道既是村民出行的必经之路，也是该村的致富之路，不仅影响村民的出行生活，农业生产物资和经营产品的进出购销也十分不便，还存在安全隐患。文章见报后，周口市委认真研究，淮阳县扶贫、交通等有关单位实地走访勘察，提出建设规划。2017年底，该道路被纳入淮阳县2018年度村内道路硬化建设规划。经过紧张施工，该道路已于8月12日实现水泥硬化铺设，修好了村民期盼已久的“小康路”，有效改善了当地农村“最后一公里”的出行状况。河南淮阳县?梁修明"
print(en_prompt(text))