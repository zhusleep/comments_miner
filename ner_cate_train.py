#-*-coding:utf-8-*-
import pandas as pd
import numpy as np
import torch, os
from data_prepare import data_manager
from spo_dataset import NERCate, get_mask_attn_pool, collate_fn_ner_cate
from spo_model import SPOModel, NerLinkCateModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, get_threshold
from pytorch_pretrained_bert import BertTokenizer, BertAdam
from sklearn.model_selection import KFold, train_test_split
import gc
import logging

from sklearn.externals import joblib

file_name = 'TRAIN/Train_reviews.csv'
file_labels = 'TRAIN/Train_labels.csv'
sentences = data_manager.read_ner_cate(filename=file_name, filelabels=file_labels)
print(len(sentences))
seed_torch(2019)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
name = 'cate'
if name =='polarity':
    id_label=4
    cate_list = data_manager.polarity
else:
    id_label=3
    cate_list = data_manager.category
for train_index, test_index in kfold.split(np.zeros(len(sentences))):
    # if round==0:
    #     round+=1
    #     continue
    train_X = [sentences[i] for i in train_index]
    dev_X = [sentences[i] for i in test_index]
    BERT_MODEL = 'bert-base-chinese'
    CASED = False
    t = BertTokenizer.from_pretrained(
        BERT_MODEL,
        do_lower_case=True,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        #    cache_dir="~/.pytorch_pretrained_bert/bert-large-uncased-vocab.txt"
    )

    train_dataset = NERCate([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in train_X], t,
                                pos1=[x[1] for x in train_X], pos2=[x[2] for x in train_X],
                                label=[x[id_label] for x in train_X], use_bert=True)
    valid_dataset = NERCate([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in dev_X], t,
                                pos1=[x[1] for x in dev_X], pos2=[x[2] for x in dev_X],
                                label=[x[id_label] for x in dev_X], use_bert=True)
    train_batch_size = 2
    valid_batch_size = 2
    model = NerLinkCateModel(vocab_size=None, init_embedding=None, encoder_size=128, dropout=0.2, num_outputs=len(cate_list), use_bert=True)
    use_cuda=True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_ner_cate, shuffle=False, batch_size=valid_batch_size)

    epochs = 2
    t_total = int(epochs*len(train_X)/train_batch_size)
    optimizer = BertAdam(model.parameters(),lr=2e-5, warmup=0.05,t_total=t_total)
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    clip = 50
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('round', round, 'epoch', epoch)
        model.train()
        train_loss = 0
        torch.cuda.manual_seed_all(epoch)
        train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn_ner_cate, shuffle=True, batch_size=train_batch_size)
        for index, X, label, pos1, pos2, length in tqdm(train_dataloader):
            #model.zero_grad()
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.to(device)
            length = length.to(device)
            mask_X = get_mask_attn_pool(X, length, pos1, pos2, is_cuda=use_cuda).to(device).type(torch.float)
            pos1 = pos1.type(torch.LongTensor).to(device)
            pos2 = pos2.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.long)

            pred = model(X, pos1, pos2, length, mask_X)
            loss = loss_fn(pred, label)
            loss.backward()
            del X, length, pred, pos1, pos2, label, mask_X
            gc.collect()
            #loss = loss_fn(pred, ner)
            optimizer.step()
            optimizer.zero_grad()
            # Clip gradients: gradients are modified in place
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            train_loss += loss.item()

        train_loss = train_loss/len(train_X)
        gc.collect()
        model.eval()
        valid_loss = 0
        pred_set = []
        label_set = []
        for index, X, label, pos1, pos2, length in tqdm(valid_dataloader):
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.to(device)
            length = length.to(device)
            #mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
            mask_X = get_mask_attn_pool(X, length, pos1, pos2, is_cuda=use_cuda).to(device).type(torch.float)
            pos1 = pos1.type(torch.LongTensor).to(device)
            pos2 = pos2.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.long)
            with torch.no_grad():
                pred = model(X, pos1, pos2, length, mask_X)
            loss = loss_fn(pred, label)
            pred_set.append(pred.cpu().numpy())
            label_set.append(label.cpu().numpy())
            valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_X)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        top_class = np.argmax(pred_set, axis=1)
        equals = top_class == label_set
        accuracy = np.mean(equals)
        print('acc', accuracy)
        print('train loss　%f, val loss %f ' % (train_loss, valid_loss))
        #INFO_THRE, thre_list = get_threshold(pred_set, label_set)
        #print(INFO_THRE)
    torch.save(model.state_dict(), 'model_ner/ner_link_round %s' % round)
    pred_vector.append(pred_set)
    round += 1
    # INFO = 'train loss %f, valid loss %f, acc %f, recall %f, f1 %f ' % (train_loss, valid_loss, INFO_THRE[0], INFO_THRE[1], INFO_THRE[2])
    # logging.info(INFO)
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
    break
result = pd.DataFrame()
result['text'] = [x[0] for x in dev_X]
result['pos1'] = [x[1] for x in dev_X]
result['pos2'] = [x[2] for x in dev_X]
result['label'] = [cate_list[x[id_label]] for x in dev_X]
result['predict'] = [cate_list[x] for x in top_class]
result.to_pickle('result/classify_%s.pkl' % name)
#pred_set
#pred_vector = np.concatenate(pred_vector, axis=0)
#np.save('result/gensim_vector_bert.npy', pred_vector)
    # 19 train loss　0.128424, val loss 0.153328, val_cos_loss 91155.319716

# 0.954
# 0.952,0.951,0.9495,0.946
# 0.9457,0.9517,0.95177 epoch==2 category drop cls
# part1 0.935 0.951 loss =  0.11
# part0 0.955 0.948 loss = 0.104

# polar
# 0.9668,0.960,0.9668 || 0.97,0.9615
# #0.970,0.975 epoch==2 0.967,0.979 keep pos2

# attn pool 0.963,