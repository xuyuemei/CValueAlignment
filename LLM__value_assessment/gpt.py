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

def augment_text_with_gpt3_with_retry(content,model_name="claude-3-opus-20240229", max_retries=100):
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





