import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
import torch
from torch import nn
import xlwt
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup


# print(test_data)
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer,text,max_length=512):
        self.data = data
        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[self.text][index]
        # print(text)
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        # print(text,labels)

        return input_ids, attention_mask

def validation(data_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, (input_ids, attention_mask) in tqdm(enumerate(data_loader, 0)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            # print(input_ids, attention_mask)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs


# text_list=["deepseekr1","glm4","qwen2.5-72b-instruct","gpt4","claude-3-opus","llama3.1-8b","DeepSeek-R1-Distill-Llama-8B","llama3.1-70b"]
text_list=["Llama","Qwen","Qwen-lora","Llama-lora"]
workbook = xlwt.Workbook()
for text in text_list:
    MODEL_NAME = '/data/qzh/val_cn/Qwen2.5-0.5B'
    NUM_LABELS = 12

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification").cuda()
    model.config.pad_token_id = tokenizer.pad_token_id


    header = pd.read_excel("/home/qzh/Value_cn/latest/Questions/News(1).xlsx", nrows=1)
    test_data = pd.read_excel("/home/qzh/Value_cn/latest/Questions/News(1).xlsx")
    # print(test_data)

    test_dataset = CustomDataset(test_data, tokenizer,text)

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 将模型移到设备上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)


    model.load_state_dict(torch.load("/data/qzh/val_cn/corevalues_ck/final_2000MulQwen_ours_asy.pt"))
    # model.to(device)


    test_outputs = validation(test_loader)
    c = np.array(test_outputs)

    rows=test_outputs
    

    print(rows)
    sheet = workbook.add_sheet(text)

    for i in range(len(rows)):
        for j in range(len(rows[i])):
            if rows[i][j]>=0.5:
                sheet.write(i, j, 1)
            else:
                sheet.write(i, j, 0)


workbook.save("test1.xls")
