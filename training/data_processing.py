from torch.utils.data import WeightedRandomSampler, DataLoader
import torch
# from Dataset import MultiLabelDataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AutoModelForSequenceClassification
from torch.utils.data import Dataset
import numpy as np

class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer,max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
    
        text = self.data['text'].loc[index]

        if 'row_variance' in self.data.columns:
            weight_label=self.data['row_variance'].loc[index]    
            labels=self.data.drop(columns=['text', 'row_variance']).loc[index].tolist()

        else:
            weight_label=1  
            labels=self.data.drop(columns=['text']).loc[index].tolist()

        labels = torch.tensor(labels, dtype=torch.float32).squeeze()
        weight_label= torch.tensor(weight_label, dtype=torch.float32).squeeze()
       
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask, labels,weight_label
    
def split_dataset(data_path,is_major=False,dimension="Prosperity"):
    data = pd.read_csv(data_path, delimiter=',', encoding='utf-8',header=0)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    data.replace(10, 1, inplace=True)
    data=data.fillna(0)#random_state=42 #random_state
    data = data.sample(frac=1, random_state=9).reset_index(drop=True)
    if is_major:
        data=data[['text',dimension]]

    train_data = data.iloc[:int(len(data) -600), :]
    val_data = data.iloc[int(len(data) -600):int(len(data)-200), :]
    test_data = data.iloc[int(len(data)-200):, :]

    # train_data = data.iloc[:int(2500*0.7), :]
    # val_data = data.iloc[int(2500*0.7):int(2500*0.9), :]
    # test_data = data.iloc[int(len(data)-200):, :]

    # 重置索引，确保索引从 0 开始
    train_data = train_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    return train_data,val_data,test_data

def get_loader(data,tokenizer,batch_size=8,if_sample=False,sample_label='富强'):

    dataset = MultiLabelDataset(data, tokenizer)

    if if_sample: #sampler
        labels=torch.tensor(data[sample_label].tolist())
        class_sample_counts = torch.bincount(labels)
        class_weights = 1.0 / class_sample_counts.float()  
        sample_weights = [class_weights[labels] for labels in labels]  
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(dataset, batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size)

