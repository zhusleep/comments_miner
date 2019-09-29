#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager
from spo_dataset import NERLINK, get_mask, collate_fn_ner_link
from spo_model import SPOModel, NerLinkModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, cal_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold, train_test_split
import logging

from sklearn.externals import joblib
import pickle

file_name = 'TRAIN/Train_laptop_reviews.csv'
file_labels = 'TRAIN/Train_laptop_labels.csv'
sentences = data_manager.read_nerlink(filename=file_name, filelabels=file_labels)
file_name = 'TRAIN/Train_makeup_reviews.csv'
file_labels = 'TRAIN/Train_makeup_labels.csv'
sentences2 = data_manager.read_nerlink(filename=file_name, filelabels=file_labels)
sentences += sentences2
test_data = pd.read_pickle('result/ner_bert_result.pkl')
test_data = test_data.reset_index().rename(columns={'index':'id'})
test_data['text'] = test_data['text'].apply(lambda x:''.join(x[1:-1]))
test_data['pos'] = test_data['pos'].apply(lambda x:[(item[0]-1,item[1]-1,item[2][1],item[2][2:]) for item in x])
test_xu = pickle.load(open('test_data_a_o_927.pkl', 'rb'))
text_df = []
pos_df = []
for item in test_xu:
    text_df.append(item['text'])
    temp = []
    if 'a' in item:
        for a_part in item['a']:
            temp.append((a_part[1], a_part[2]-1, 'A', a_part[5]))
    if 'o' in item:
        for o_part in item['o']:
            temp.append((o_part[1], o_part[2]-1, 'O', o_part[5]))
    temp = sorted(temp, key=lambda x: x[0], reverse=False)
    pos_df.append(temp)
test_data = pd.DataFrame()
test_data['text'] = text_df
test_data['pos'] = pos_df
test_data = test_data.reset_index().rename(columns={'index':'id'})


span = test_data['pos'].to_dict()
idToText = test_data['text'].to_dict()
# find potential A-O connection
training = []
for id in span:
    for index, item in enumerate(span[id]):
        if item[2] == 'A':

            for other_index in [index - 2, index - 1, index + 1, index + 2]:
                if other_index >= 0 and other_index <= len(span[id])-1 and span[id][other_index][2] != 'A':# and span[id][other_index][3]==span[id][index][3]:
                    [a, b] = sorted([index, other_index])
                    if b - a == 2 and span[id][b][3] != span[id][a + 1][3]:
                        continue
                    sen = idToText[id]
                    # label 0 temepory
                    training.append((sen, span[id][a], span[id][b], 0, abs(other_index - index) - 1, id))

print(len(sentences), sum([x[3] for x in sentences]))
print(len(training))
seed_torch(2019)

train_X = sentences
dev_X = training
BERT_MODEL = 'bert-base-chinese'
CASED = False
t = BertTokenizer.from_pretrained(
    BERT_MODEL,
    do_lower_case=True,
    never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
)
train_dataset = NERLINK([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in train_X], t,
                            pos1=[x[1] for x in train_X], pos2=[x[2] for x in train_X],
                            label=[x[3] for x in train_X], use_bert=True, gap=[x[4] for x in train_X])
valid_dataset = NERLINK([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in dev_X], t,
                            pos1=[x[1] for x in dev_X], pos2=[x[2] for x in dev_X],
                            label=None, use_bert=True, gap=[x[4] for x in dev_X])
train_batch_size = 20
valid_batch_size = 20
model = NerLinkModel(vocab_size=None, init_embedding=None, encoder_size=128, dropout=0.2, use_bert=True)
use_cuda=True
if use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
model.to(device)
valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_ner_link, shuffle=False, batch_size=valid_batch_size)

epochs = 3
t_total = int(epochs*len(train_X)/train_batch_size)
# optimizer = BertAdam([
#                 {'params': model.LSTM.parameters()},
#                 {'params': model.hidden.parameters()},
#                 {'params': model.classify.parameters()},
#                 {'params': model.span_extractor.parameters()},
#                 {'params': model.bert.parameters(), 'lr': 2e-5}
#             ],  lr=1e-3, warmup=0.05,t_total=t_total)
optimizer = BertAdam([
    {'params': model.parameters(), 'lr': 2e-5}
], lr=2e-5, warmup=0.05, t_total=t_total)
#optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
clip = 50
loss_fn = nn.BCEWithLogitsLoss()
for epoch in range(epochs):
    print('round', round, 'epoch', epoch)
    model.train()
    train_loss = 0
    torch.cuda.manual_seed_all(epoch)
    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_ner_link, shuffle=True, batch_size=train_batch_size)
    for index, X, label, pos1, pos2, length, numerical_f in tqdm(train_dataloader):
        #model.zero_grad()
        X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
        X = X.to(device)
        length = length.to(device)
        #mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
        pos1 = pos1.type(torch.LongTensor).to(device)
        pos2 = pos2.type(torch.LongTensor).to(device)
        label = label.to(device).type(torch.float).view(-1,1)
        numerical_f = numerical_f.to(device)

        pred = model(X, pos1, pos2, length, numerical_f)
        loss = loss_fn(pred, label)
        loss.backward()

        #loss = loss_fn(pred, ner)
        optimizer.step()
        optimizer.zero_grad()
        # Clip gradients: gradients are modified in place
        # nn.utils.clip_grad_norm_(model.parameters(), clip)
        train_loss += loss.item()

    train_loss = train_loss/len(train_X)

torch.save(model.state_dict(), 'model_ner/ner_link.pth')
model.load_state_dict(torch.load('model_ner/ner_link.pth', map_location=torch.device('cuda')))

model.eval()
valid_loss = 0
pred_set = []
for index, X, pos1, pos2, length, numerical_f in tqdm(valid_dataloader):
    X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
    X = X.to(device)
    length = length.to(device)
    #mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
    pos1 = pos1.type(torch.LongTensor).to(device)
    pos2 = pos2.type(torch.LongTensor).to(device)
    numerical_f = numerical_f.to(device)
    with torch.no_grad():
        pred = model(X, pos1, pos2, length, numerical_f)
    pred_set.append(pred.cpu().numpy())
    valid_loss += loss.item()

pred_set = np.concatenate(pred_set, axis=0)

for i in range(len(pred_set)):
    if valid_dataset.type_error[i] == 0:
        pred_set[i, 0] = 0

AO_link = {}
thre = cal_threshold(pred_set)
true_link = 0
for index, item in enumerate(dev_X):

    if pred_set[index,0]>=thre:
        true_link +=1
        if item[5] not in AO_link:
            AO_link[item[5]] = [(item[1], item[2], pred_set[index])]
        else:
            AO_link[item[5]].append((item[1],item[2], pred_set[index]))
print('old true link', true_link)


def generate_result(pos, id):
    result = []
    for ner in pos:
        flag = False
        if id in AO_link:
            for item in AO_link[id]:

                if (ner[0],ner[1]) in [(item[0][0],item[0][1]),(item[1][0],item[1][1])]:
                    flag = 1
                    item = sorted(item,key=lambda x: x[2])
                    if (item[0], item[1], item[0][3]) not in result:
                        result.append((item[0], item[1], item[0][3]))
        if not flag:
            if ner[2]=='A':
                if (ner, None, ner[3]) not in result:
                    result.append((ner, None, ner[3]))
            elif (None, ner, ner[3]) not in result:
                result.append((None, ner, ner[3]))
    return result

print(len(test_data))
test_data['result'] = test_data.apply(lambda x: generate_result(x.pos, x.id), axis=1)
test_data.to_pickle('result/ner_link.pkl')

# old link 6150