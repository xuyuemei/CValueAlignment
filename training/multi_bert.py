import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch
from torch import nn
# import warnings
# warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
# from Dataset import MultiLabelDataset
# import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification,AutoTokenizer,AutoModelForSequenceClassification
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import get_linear_schedule_with_warmup
np.set_printoptions(threshold=np.inf)
from data_processing import *
from loss import *

# data_xls = pd.read_excel('/home/qzh/Value_cn/humanlabelnew.xlsx')
# data_xls.to_csv('/home/qzh/Value_cn/humanlabelnew.csv', encoding='utf-8')
# data_xls = pd.read_excel('/home/qzh/Value_cn/dataset/humanlabel_label.xlsx')
# data_xls.to_csv('/home/qzh/Value_cn/humanlabel_new.csv', encoding='utf-8')


words_to_keep=["Prosperity", "Democracy", "Civility", "Harmony", "Freedom", "Equality", "Justice", "Rule of Law","Patriotism", "Dedication", "Integrity", "Friendliness"]
dimension=words_to_keep[0]
labeler_type="" 
data_path="/home/qzh/Value_cn/latest/dataset/3000/llmlabeled3000_rj.csv"
golden_path="/home/qzh/Value_cn/latest/dataset/2000/humanlabeled2000.csv"
is_weight=False
loss_type="asy"  # "bf" "org" "asy" "mf"

# dimension="human"
check_path="/data/qzh/val_cn/corevalues_ck/qwen_asy/3000"
# check_path="/data/qzh/val_cn/corevalues_ck/24labelsall"+loss_type
# check_path="/data/qzh/val_cn/corevalues_ck/humanmajor/asy/24labels"
# check_path="/data/qzh/val_cn/corevalues_ck/llmlabels/org/24labels"
print(check_path)
NUM_LABELS=12
MODEL_NAME = '/data/qzh/val_cn/Qwen2.5-0.5B'#Qwen2.5-1.5B Llama-3.2-1B

is_major= False
is_train= False
NUM_EPOCHS = 15
best_val_score = 0.0
patience = 5  # Number of epochs to wait for improvement
num_epochs_no_improvement = 0
LEARNING_RATE = 1e-5
Threshold=0.7


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# tokenizer.pre_tokenizer = Whitespace()

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, problem_type="multi_label_classification")

train_data, val_data, test_data = split_dataset(data_path)
golden_train_data , golden_val_data, golden_test_data = split_dataset(golden_path, dimension=dimension)
# print(test_data,golden_test_data)
train_loader = get_loader(train_data, tokenizer, batch_size=8)
val_loader = get_loader(val_data, tokenizer, batch_size=8)
# if is_major:
#     test_loader = get_loader(golden_test_data, tokenizer,batch_size=8)
# else:
#     test_loader = get_loader(test_data, tokenizer,batch_size=8)
test_loader = get_loader(golden_test_data, tokenizer, batch_size=8)
# test_loader = get_loader(test_data, tokenizer,batch_size=8)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


model=model.cuda()
model.config.pad_token_id = tokenizer.pad_token_id

if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
    

model = model.to(device)

if loss_type == "bf":
    criterion = BinaryFocalLoss(gamma=2, reduction='mean')
elif loss_type == "org":
    criterion_weight = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
elif loss_type == "asy":
    criterion = AsymmetricLossOptimized()
elif loss_type == "ct":
    criterion = MultiLabelContrastiveFocalLoss()

# 初始化optimizer和scheduler
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                            num_training_steps=total_steps)


# 定义训练器
def train(epoch):
    model.train()
    for _, (input_ids, attention_mask, labels,weight_label) in tqdm(enumerate(train_loader, 0)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        targets = labels.to(device)
        weight_label = weight_label.to(device)

        outputs = model(input_ids, attention_mask=attention_mask).logits
        optimizer.zero_grad()
        # print(outputs, targets)
        loss = criterion(outputs, targets)
        
        # print("loss:",loss)
        # loss=loss.mean()
        # loss = criterion_weight(outputs, targets)
        # print(loss1,loss2)
        # loss=torch.mean(loss, dim=1)
        # loss=loss*weight_label
        # loss=loss.mean()
        # print(loss1,loss2)
        # print("weight_label",labels,weight_label,loss)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()
        # 更新学习率 (Learning rate warm-up)
        scheduler.step()


# 定义验证函数
def validation(data_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, (input_ids, attention_mask, labels,weight_label) in tqdm(enumerate(data_loader, 0)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = labels.to(device)
            weight_label=weight_label.to(device)
            # print(input_ids, attention_mask)
             # print("loss:",loss.item())
            outputs = model(input_ids, attention_mask=attention_mask).logits
            # print(outputs, targets)
            loss = criterion(outputs, targets)
            
            # print("loss:",loss)
            # loss=loss.mean()
            # loss = criterion_weight(outputs, targets)
            # loss=torch.mean(loss, dim=1)
            # loss=loss*weight_label
            # loss=loss.mean()
            # print(loss1,loss2)
            if _ % 5000 == 0:
                print(f'Validation Loss: {loss.item()}')
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

def evaluate_metrics(targets, outputs,is_major=False):
    targets = np.array(targets)
    outputs = np.array(outputs)
    # 
    outputs = (outputs > Threshold).astype(int)
    # print(targets,outputs)
    
    num_ones_in_targets = np.sum(targets == 1)
    num_ones_in_outputs = np.sum(outputs == 1)

    print(f"targets 中为 1 的个数: {num_ones_in_targets}")
    print(f"outputs 中为 1 的个数: {num_ones_in_outputs}")
    # print(targets, outputs)
    overall_accuracy = accuracy_score(targets, outputs)
    overall_f1 = f1_score(targets, outputs, average='micro')
    overall_precision = precision_score(targets, outputs, average='micro')
    overall_recall = recall_score(targets, outputs, average='micro')

    class_accuracies = {}
    class_f1s={}
    class_precisions={}
    class_recalls={}
    if is_major==False:
        for i, label in enumerate(list(range(NUM_LABELS))):
            class_targets = targets[:, i]
            class_outputs = outputs[:, i]

            class_accuracy = accuracy_score(class_targets, class_outputs)
            class_accuracies[label] = class_accuracy

            class_f1 = f1_score(class_targets, class_outputs)
            class_f1s[label] = class_f1

            class_precision = precision_score(class_targets, class_outputs)
            class_precisions[label] = class_precision

            class_recall = recall_score(class_targets, class_outputs)
            class_recalls[label] = class_recall
    else:
        print("Major Test:\n")
        label="Major"
        
        class_targets = targets
        class_outputs = outputs

        class_accuracy = accuracy_score(class_targets, class_outputs)
        class_accuracies[label] = class_accuracy

        class_f1 = f1_score(class_targets, class_outputs)
        class_f1s[label] = class_f1

        class_precision = precision_score(class_targets, class_outputs)
        class_precisions[label] = class_precision

        class_recall = recall_score(class_targets, class_outputs)
        class_recalls[label] = class_recall



    return overall_accuracy, overall_f1, overall_precision, overall_recall, class_accuracies,class_f1s,class_precisions,class_recalls


def test(data_loader):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, (input_ids, attention_mask, labels,_) in tqdm(enumerate(data_loader, 0)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            targets = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask).logits
            # print(targets,outputs)

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    # print(len(fin_outputs), len(fin_targets))

    return fin_outputs, fin_targets

if is_train:
    for epoch in range(NUM_EPOCHS):
        train(epoch)

        # 验证模型并计算评价指标
        val_outputs, val_targets = validation(val_loader)
        print(val_outputs, val_targets)
        val_accuracy, val_f1, val_precision, val_recall, val_class_accuracies,val_class_f1s,val_class_precisions,val_class_recalls = evaluate_metrics(val_targets, val_outputs)
        # print(
        #     f"Validation Accuracy: {val_accuracy:.4f}, F1-score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        # for label, accuracy in val_class_accuracies.items():
        #     print(f"Val {label} Accuracy: {accuracy:.4f}")
        # 保存在验证集上效果最好的模型
        if val_f1 > best_val_score:
            best_val_score = val_f1
            torch.save(model.state_dict(), check_path+".pt")
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1

        # 判断是否提前停止训练
        if num_epochs_no_improvement >= patience:
            print("Early stopping triggered. Training stopped.")
            break

print("loding from...",check_path)
model.load_state_dict(torch.load(check_path+".pt"))
model.to(device)

# 在测试集上进行评估
if is_major:
    test_outputs, test_targets = test(test_loader)
else:
    test_outputs, test_targets = validation(test_loader)

print(test_outputs, test_targets)
test_accuracy, test_f1, test_precision, test_recall, test_class_accuracies ,test_class_f1s,test_class_precisions,test_class_recalls= evaluate_metrics(test_targets, test_outputs,is_major=is_major)

print(
    f"Test Overall Accuracy: {test_accuracy:.4f}, F1-score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

for label, accuracy in test_class_accuracies.items():
    print(accuracy, end=",")
print("")
for label, f1 in test_class_f1s.items():
    # print(f"Test {label} f1: {f1:.4f}")
    print(f1, end=",")
print("")
for label, precision in test_class_precisions.items():
    # print(f"Test {label} precisions: {precision:.4f}")
    print(precision, end=",")
print("")
for label, recall in test_class_recalls.items():
    # print(f"Test {label} recalls: {recall:.4f}")
    print(recall, end=",")
