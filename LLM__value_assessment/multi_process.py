# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import collections
import pandas as pd
from generate_que import *

start_time = time.time()

input_data = []

df = pd.read_excel("quest500.xlsx")
input_data=df['infer_result'].to_list()
# print(input_data)
# output_data = collections.defaultdict(int)

with ProcessPoolExecutor(max_workers=64) as executor:
    ## 同步调用.submit之后直接.result（一个进程执行完才能下一个进程）
    # output_data = [executor.submit(predict, prompt.format(query)).result() for query in input_data]

    # 异步调用（多进程并发执行）
    futures = [executor.submit(ask, query) for query in input_data]
    query2res = collections.defaultdict(int) # 因为异步等待结果，返回的顺序是不定的，所以记录一下进程和输入数据的对应
    # 异步等待结果（返回顺序和原数据顺序可能不一致） ，直接predict函数里返回结果？
    for job in as_completed(futures):
        query,res = job.result(timeout=None)  # 默认timeout=None，不限时间等待结果
        query2res[query] = res
        time.sleep(3)  # 为了避免超过OpenAI API的速率限制，每次预测之间间隔1秒

# from openpyxl import Workbook
# from openpyxl import load_workbook

# # 创建一个新的 Excel 工作簿
# wb = load_workbook("llama3.1-70b-instruct.xlsx")
# ws = wb.active

# for row_index, query in enumerate(input_data, start=1):
#     # if row_index <= 40:
#     #     continue
#     output_data=ask(query)
#     ws.cell(row=row_index, column=1, value=output_data[0])
#     ws.cell(row=row_index, column=2, value=output_data[1])
#     wb.save("llama3.1-70b-instruct.xlsx")
#     print(f"已写入第 {row_index} 行并保存：{output_data}")
    # break

end_time = time.time()
total_run_time = round(end_time-start_time, 3)
print('Total_run_time: {} s'.format(total_run_time))
print(query2res)
df = pd.DataFrame({'query': list(query2res.keys()), 'infer_result': list(query2res.values())})
df.to_csv('claude.csv')
