import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager
from spo_dataset import SPO_BERT, get_mask, collate_fn
from spo_model import SPOModel, SPO_Model_Bert
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, cal_ner_result
from pytorch_pretrained_bert import BertTokenizer, BertAdam
import logging
import time

current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)


seed_torch(2019)
file_namne = 'TRAIN/Train_reviews.csv'
file_labels = 'TRAIN/Train_labels.csv'
sentences, labels = data_manager.parseData(filename=file_namne, filelabels=file_labels)
# test set
dev_X = pd.read_csv('test/Test_reviews.csv')['Reviews']

## deal for bert
train_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in sentences]
dev_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in dev_X]
train_ner = [[0]+list(temp)+[0]for temp in labels]
result = cal_ner_result(train_ner, data_manager)

BERT_MODEL = 'bert-base-chinese'
CASED = False
t = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True,
    never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
#    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)


train_dataset = SPO_BERT(train_X, t,  ner=train_ner)
valid_dataset = SPO_BERT(dev_X, t, ner=None)

batch_size = 10

model = SPO_Model_Bert(encoder_size=128, dropout=0.5, num_tags=len(data_manager.ner_list))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn, shuffle=False, batch_size=batch_size)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

epochs = 3
t_total = int(epochs*len(train_X)/batch_size)
optimizer = BertAdam([
                {'params': model.LSTM.parameters()},
                {'params': model.hidden.parameters()},
                {'params': model.NER.parameters()},
                {'params': model.crf_model.parameters()},
                {'params': model.bert.parameters(), 'lr': 2e-5}
            ],  lr=1e-3, warmup=0.05, t_total=t_total)
clip = 50

#model.load_state_dict(torch.load('model_ner/ner_bert_predict.pth'))

for epoch in range(epochs):
    model.train()
    train_loss = 0
    torch.cuda.manual_seed_all(epoch)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=batch_size)
    for index, X, ner, length in tqdm(train_dataloader):
        # for eval ###################################

        # ##########################################
        #model.zero_grad()
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.cuda()
        length = length.cuda()
        #ner = ner.type(torch.float).cuda()
        mask_X = get_mask(X, length, is_cuda=True).cuda()
        ner = nn.utils.rnn.pad_sequence(ner, batch_first=True).type(torch.LongTensor)
        ner = ner.cuda()

        loss = model.cal_loss(X, mask_X, length, label=ner)
        loss.backward()

        #loss = loss_fn(pred, ner)
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        optimizer.zero_grad()
        # Clip gradients: gradients are modified in place
        train_loss += loss.item()
    train_loss = train_loss/len(train_X)
    #if epoch==1:break

torch.save(model.state_dict(), 'model_ner/ner_bert_predict.pth')
#
model.eval()
pred_set = []
for index, X, length in tqdm(valid_dataloader):
    X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
    X = X.cuda()
    length = length.cuda()
    mask_X = get_mask(X, length, is_cuda=True).cuda()
    with torch.no_grad():
        pred = model(X, mask_X, length)
    for i, item in enumerate(pred):
        pred_set.append(item[0:length.cpu().numpy()[i]])

# stop
result = cal_ner_result(pred_set, data_manager)

# 正负样本分析
dev = pd.DataFrame()
dev['text'] = dev_X
pred_mention = []
pred_pos = []
for index, row in dev.iterrows():
    temp_mention = []
    for item in result[index]:
        temp_mention.append((''.join(row['text'][item[0]:item[1]+1]), item[-1]))
    pred_mention.append(temp_mention)

dev['pred'] = pred_mention
dev['pos'] = result
dev.to_pickle('result/ner_bert_result.pkl')


