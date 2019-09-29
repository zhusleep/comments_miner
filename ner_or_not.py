print('33333')
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager
from spo_dataset import SPO_BERT_nerornot, get_mask, collate_fn_nerornot
from spo_model import SPOModel, NerOrNot
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, cal_ner_result, get_threshold
from pytorch_pretrained_bert import BertTokenizer, BertAdam
import logging
import time
from sklearn.model_selection import KFold
from pathlib import Path
from pytorch_transformers import BertTokenizer, BertModel

import gc

current_name = 'log/%s.txt' % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
logging.basicConfig(filename=current_name,
                    filemode='w',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.INFO)
seed_torch(2019)
file_namne = 'TRAIN/Train_laptop_reviews.csv'
file_labels = 'TRAIN/Train_laptop_labels.csv'
sentences, labels = data_manager.parseData_nerornot(filename=file_namne, filelabels=file_labels)
file_namne = 'TRAIN/Train_makeup_reviews.csv'
file_labels = 'TRAIN/Train_makeup_labels.csv'
sentences2, label2 = data_manager.parseData_nerornot(filename=file_namne, filelabels=file_labels)

print(sum(labels),len(labels))
weight_type=['default','ernie1','wwm']
weight_type = 'default'

VOCAB_PATH = Path('bert/%s' % weight_type)
PRETRAIN_PATH = Path('bert/%s' % weight_type)

print(PRETRAIN_PATH)
# t = BertTokenizer.from_pretrained('bert/ernie1/')
print('end')
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
dev_all = []
for train_index, test_index in kfold.split(np.zeros(len(sentences))):
    ## deal for bert
    # if round < 4:
    #     round += 1
    #     continue

    train_X = [sentences[i] for i in train_index]#+sentences2
    train_ner = [labels[i] for i in train_index]#+label2
    dev_X = [sentences[i] for i in test_index]
    dev_ner = [labels[i] for i in test_index]

    train_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in train_X]
    dev_X = [['[CLS]']+list(temp)+['[SEP]'] for temp in dev_X]
    BERT_MODEL = 'bert-base-chinese'
    CASED = False
    t = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=True
        #never_split = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
    #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
    )

    train_dataset = SPO_BERT_nerornot(train_X, t,  ner=train_ner)
    valid_dataset = SPO_BERT_nerornot(dev_X, t, ner=dev_ner)

    batch_size = 20

    model = NerOrNot(encoder_size=128, dropout=0.5, pretrain_path=PRETRAIN_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_nerornot, shuffle=True, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_nerornot, shuffle=False, batch_size=batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    epochs = 4
    t_total = int(epochs*len(train_X)/batch_size)
    optimizer = BertAdam([

                    {'params': model.bert.parameters(), 'lr': 2e-5}
                ],  lr=1e-3, warmup=0.05, t_total=t_total)
    clip = 50

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        torch.cuda.manual_seed_all(epoch)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_nerornot, shuffle=True, batch_size=batch_size)

        #model.load_state_dict(torch.load('model_ner/ner_bert.pth'))
        for index, X, ner1, length in tqdm(train_dataloader):
            #model.zero_grad()
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.cuda()
            length = length.cuda()
            ner1 = ner1.to(device).type(torch.float).view(-1, 1)

            mask_X = get_mask(X, length, is_cuda=True).cuda()
            loss = model.cal_loss(X, label1=ner1)
            loss.backward()
            #print('after', model.NER1.weight.grad)

            #loss = loss_fn(pred, ner)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            # Clip gradients: gradients are modified in place
            train_loss += loss.item()
            gc.collect()
            #break

        train_loss = train_loss/len(train_X)

        model.eval()
        valid_loss = 0
        pred_set = []
        label_set = []
        for index, X, ner1, length in tqdm(valid_dataloader):
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.cuda()
            length = length.cuda()
            mask_X = get_mask(X, length, is_cuda=True).cuda()
            ner1 = ner1.to(device).type(torch.float).view(-1, 1)
            with torch.no_grad():
                pred = model(X)
                loss = model.cal_loss(X, label1=ner1)
            for i, item in enumerate(pred):
                pred_set.append(item.cpu().numpy())
            #pred_set.extend(pred)
            for item in ner1:
                label_set.append(item.cpu().numpy())
            valid_loss += loss.item()
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        valid_loss = valid_loss/len(dev_X)
        gc.collect()
        #print(pred_set.shape, label_set.shape)
        pred_set = pred_set.reshape(-1,1)
        label_set = label_set.reshape(-1,1)

        INFO_THRE, thre_list = get_threshold(pred_set, label_set)
        #INFO = 'epoch %d, train loss %f, valid loss %f, acc %f, recall %f, f1 %f ' % (epoch, train_loss, valid_loss,acc,recall,f1)
        #logging.info(INFO)
        print(INFO_THRE)
        # if epoch == 3:
        #     break


    #break
dev_all = pd.concat(dev_all,axis=0)
dev_all.to_csv('result/analysis.csv', sep='\t')

# 0912 v1 0.842 0.828 0.840 0.835 lr=1e-3
# 0912 v2 0.849 0.836 0.848 0.836 lr=5e-4
# 0912 v3 0.855 0.831 0.841 5e-4 epoch=6
# 五折　0.860 0.846 0.832
# no shuffle 0.850 0.831 0.850
# 0.852 0.834/0.851
# ernie1.0 0.845 0.818 0.844 0.833 0.836
# 0.849 0.8240.848 0.837
#  0.853 0.837 0.863 0.851 0.837
# crf 0.848 0.855 0.857 0.855 0.833
# ernie 0.851,0.852,0.855,0.853 0.840
