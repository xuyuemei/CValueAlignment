import openai  #openai==0.28.1
import time
import socket
import urllib3.exceptions
import requests.exceptions
import openai.error
import jsonlines
import os
import math
import os
# from openai import OpenAI #openai==1.2.0
import requests
import time
import json
import time
# from zhipuai import ZhipuAI
# from zhipuai.core._errors import APIRequestFailedError
import pandas as pd
api_key = "YOUR_API_KEY"
BASE_URL = "https://api.zhizengzeng.com/v1/"
# https://flag.smarttrot.com/v1/
# api_key="YOUR_API_KEY"
# BASE_URL="https://vip.apiyi.com/v1"
def augment_text_with_gpt3_with_retry(content,model_name="gpt-4o", max_retries=100):
    openai.api_key = api_key
    openai.api_base= BASE_URL
    for _ in range(max_retries):
        try:
            # print("Prompt/n",content)
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": content}
                ],
                temperature=1,
                request_timeout=60,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5,
                stop=None,
                max_tokens=4096
            )
            if 'choices' in response:
                return response.choices[0].message.content
            else:
                print("字典不包含 'choices' 键",response)
                return None

            
        except (socket.timeout, urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout,
                openai.error.Timeout, openai.error.ServiceUnavailableError,
                openai.error.RateLimitError, openai.error.APIError):
            # Handle network error and retry
            print("Socket timeout, retrying...")
            time.sleep(30)  # Add a short delay before retrying
    # If max_retries is reached, return an error message
    return "Network error: Max retries reached"


from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import collections


# 定义预测函数
def predict(params):
    prompt, query,model_name = params
    prompt = prompt.format(query)

    openai.api_key = api_key
    openai.api_base="https://flag.smarttrot.com/v1/"

    retry_count = 100
    retry_interval = 1
    for _ in range(retry_count):
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
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


