from torch.utils.data import Dataset
import torch

class ValueDataset(Dataset):
    def __init__(self, data, tokenizer, max_len,val_idx,text_idx):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.val_idx=val_idx
        self.text_idx=text_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index, self.text_idx]
        # print("属地化",self.data, self.text_idx,"是对哦暗红色的",text)
        target = self.data.iloc[index,self.val_idx]
        # print("属地化",self.val_idx,target)
        # print(target)
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()

        return {
            'input_ids': torch.tensor(input_ids).clone().detach().long(),
            'attention_mask': torch.tensor(attention_mask).clone().detach().long(),
            'token_type_ids': torch.tensor(token_type_ids).clone().detach().long(),
            'targets': torch.tensor(target).clone().detach().long(),
        }
    
class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, labels,weight_label,max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = labels
        self.weight_label = weight_label
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'][index]
        labels=self.labels
        weight_label=self.data[self.weight_label][index]     
        labels_batch = {k: self.data[k] for k in self.data.keys() if k in labels}
        # print(labels_batch)
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros(len(labels))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[idx] = labels_batch[label][index]

        labels = labels_matrix.tolist()
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
        # print(text,labels,weight_label)

        return input_ids, attention_mask, labels,weight_label