#!/usr/bin/env python
# coding: utf-8

# # import package

# In[1]:


import json
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import datetime
from IPython.display import clear_output


# In[2]:


# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset, random_split


# In[3]:


# bert
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
clear_output()


# # REPRODUCIBILITY

# In[4]:


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


# # parameter

# In[5]:


DEVICE = "cuda:1"

# bert
PRETRAINED_MODEL_NAME = "bert-base-chinese"
MAX_LENGTH = 100

# train
LEARNING_RATE = 0.00001
EPOCHS = 2


# # read data 

# In[6]:


temp_ = "./data/%s"
with open(temp_ % "X_train.json") as json_file:
    X_train = json.load(json_file)
    
with open(temp_ % "X_test.json") as json_file:
    X_test = json.load(json_file)
    
with open(temp_ % "y_train.json") as json_file:
    y_train = json.load(json_file)
    
with open(temp_ % "y_test.json") as json_file:
    y_test = json.load(json_file)


# In[7]:


tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)


# In[8]:


# X_train, y_train = X_train[:500], y_train[:500]
# X_test, y_test = X_test[:500], y_test[:500]


# # create loader

# In[9]:


def create_data_loader(X, y, batch_size_):
    
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    buf = [tokenizer.encode_plus(i, do_lower_case = False, add_special_tokens = True, max_length = MAX_LENGTH, pad_to_max_length = True) for i in tqdm(X)]   
    input_ids = torch.LongTensor( [i['input_ids'] for i in buf] )
    token_type_ids = torch.LongTensor( [i['token_type_ids'] for i in buf])
    attention_mask = torch.LongTensor( [i['attention_mask'] for i in buf])

    #label = torch.FloatTensor(y)
    label = torch.tensor(y)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size_, shuffle = True)

    return(loader)


# In[10]:


train_loader = create_data_loader(X_train, y_train, 128)
test_loader = create_data_loader(X_test, y_test, 128)


# # create model

# In[11]:


model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels = 10)
model.to(DEVICE)
clear_output()


# In[12]:


optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)


# # def acc

# In[13]:


def get_predictions(model, dataloader, compute_acc = False):
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in dataloader:
            data = [t.to(DEVICE) for t in data]

            tokens_tensors, segments_tensors, masks_tensors = data[:3]
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc

    return predictions


# In[14]:


# get_predictions(model = model, dataloader = train_loader, compute_acc = True)


# # START TRAIN

# In[15]:


print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
for epoch in range(EPOCHS):
    running_loss = 0.0

    for data in train_loader:    
        input_ids, token_type_ids, attention_mask, labels = [t.to(DEVICE) for t in data]

        optimizer.zero_grad()

         # forward pass
        outputs = model(input_ids = input_ids, 
                        token_type_ids = token_type_ids, 
                        attention_mask = attention_mask, 
                        labels = labels)          

        loss = outputs[0]
        # backward
        loss.backward()
        optimizer.step()
        
        running_loss = running_loss + loss.item()

    print("\n===EPOCH %d/%d==="% (epoch+1, EPOCHS)) 
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    _, acc = get_predictions(model, train_loader, compute_acc=True)
    print("train_acc: %.3f" % acc)
    
    _, acc = get_predictions(model, test_loader, compute_acc=True)
    print("test_acc: %.3f" % acc)


# In[ ]:





# # for upload

# In[22]:


def create_data_loader(X, y, batch_size_):
    
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    buf = [tokenizer.encode_plus(i, do_lower_case = False, add_special_tokens = True, max_length = MAX_LENGTH, pad_to_max_length = True) for i in tqdm(X)]   
    input_ids = torch.LongTensor( [i['input_ids'] for i in buf] )
    token_type_ids = torch.LongTensor( [i['token_type_ids'] for i in buf])
    attention_mask = torch.LongTensor( [i['attention_mask'] for i in buf])

    #label = torch.FloatTensor(y)
    label = torch.tensor(y)

    dataset = TensorDataset(input_ids, token_type_ids, attention_mask, label)
    loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size_, shuffle = False)

    return(loader)


# In[23]:


with open(temp_ % "X_upload.json") as json_file:
    X_upload = json.load(json_file)

upload_loader = create_data_loader( X_upload, y_train[:59908], 128)


# In[24]:


predict = get_predictions(model, upload_loader, compute_acc = False)


# In[25]:


predict.cpu().numpy() 


# # for format

# In[26]:


data = pd.read_csv("./raw_data/sample.csv")


# In[27]:


data.label = predict.cpu().numpy() 


# In[28]:


data.to_csv("result.csv", index = False)


# In[ ]:




