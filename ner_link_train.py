#-*-coding:utf-8-*-
import pandas as pd
import numpy as np

import torch, os
from data_prepare import data_manager
from spo_dataset import NERLINK, get_mask, collate_fn_ner_link, get_mask_pos
from spo_model import SPOModel, NerLinkModel
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm as tqdm
import torch.nn as nn
from utils import seed_torch, read_data, load_glove, calc_f1, get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold, train_test_split
import logging

from sklearn.externals import joblib

file_name = 'TRAIN/Train_laptop_reviews.csv'
file_labels = 'TRAIN/Train_laptop_labels.csv'
sentences = data_manager.read_nerlink(filename=file_name, filelabels=file_labels)
file_name = 'TRAIN/Train_makeup_reviews.csv'
file_labels = 'TRAIN/Train_makeup_labels.csv'
sentences2 = data_manager.read_nerlink(filename=file_name, filelabels=file_labels)
print(len(sentences),sum([x[3] for x in sentences]))
seed_torch(2019)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
#sentences = sentences2
for train_index, test_index in kfold.split(np.zeros(len(sentences))):
    # print(round)
    # if round<2:
    #     round +=1
    #     continue

    train_X = [sentences[i] for i in train_index]#+sentences2
    dev_X = [sentences[i] for i in test_index]
    #train_X = sentences2
    #dev_X = sentences
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
                                label=[x[3] for x in dev_X], use_bert=True, gap=[x[4] for x in dev_X])

    # print(valid_dataset.type_error)
    train_batch_size = 20
    valid_batch_size = 20
    model = NerLinkModel(vocab_size=None, init_embedding=None, encoder_size=128, dropout=0.2, use_bert=True)
    use_cuda = True
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=collate_fn_ner_link, shuffle=False, batch_size=valid_batch_size)

    epochs = 4
    t_total = int(epochs*len(train_X)/train_batch_size)
    # optimizer = BertAdam(model.parameters(),  lr=1e-5, warmup=0.05, t_total=t_total)
    optimizer = BertAdam([
                    {'params': model.parameters(), 'lr': 2e-5}
                ],  lr=2e-5, warmup=0.05, t_total=t_total)
    #optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    clip = 50
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
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
            mask1, mask2 = get_mask_pos(X, length, pos1, pos2, is_cuda=use_cuda)
            mask1, mask2 = mask1.to(device).type(torch.float), mask2.to(device).type(torch.float)

            pos1 = pos1.type(torch.LongTensor).to(device)
            pos2 = pos2.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.float).view(-1, 1)
            numerical_f = numerical_f.to(device)

            pred = model(X, pos1, pos2, length, numerical_f, mask1, mask2)
            loss = loss_fn(pred, label)
            loss.backward()

            #loss = loss_fn(pred, ner)
            optimizer.step()
            optimizer.zero_grad()
            # Clip gradients: gradients are modified in place
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            train_loss += loss.item()
            #break
        train_loss = train_loss/len(train_X)

        model.eval()
        valid_loss = 0
        valid_cosloss = 0
        pred_set = []
        label_set = []
        for index, X, label, pos1, pos2, length, numerical_f in valid_dataloader:
            X = nn.utils.rnn.pad_sequence(X, batch_first=True).type(torch.LongTensor)
            X = X.to(device)
            length = length.to(device)
            #mask_X = get_mask(X, length, is_cuda=use_cuda).to(device)
            mask1, mask2 = get_mask_pos(X, length, pos1, pos2, is_cuda=use_cuda)
            mask1, mask2 = mask1.to(device).type(torch.float), mask2.to(device).type(torch.float)
            pos1 = pos1.type(torch.LongTensor).to(device)
            pos2 = pos2.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.float).view(-1, 1)
            numerical_f = numerical_f.to(device)
            with torch.no_grad():
                pred = model(X, pos1, pos2, length, numerical_f, mask1, mask2)
            loss = loss_fn(pred, label)
            pred_set.append(pred.cpu().numpy())
            label_set.append(label.cpu().numpy())
            valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_X)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        # for i in range(len(pred_set)):
        #     if valid_dataset.type_error[i]==0:
        #         pred_set[i,0] = 0
        # top_class = np.argmax(pred_set, axis=1)
        # equals = top_class == label_set
        # accuracy = np.mean(equals)
        # print('acc', accuracy)
        k = np.array(valid_dataset.gap)

        INFO_THRE, thre_list = get_threshold(pred_set[k == 1], label_set[k == 1])
        INFO_THRE, thre_list = get_threshold(pred_set[k == 0], label_set[k == 0])
        INFO_THRE, thre_list = get_threshold(pred_set, label_set)

        print('round', round, 'epoch', epoch,'train loss　%f, val loss %f ' % (train_loss, valid_loss), INFO_THRE)

    #torch.save(model.state_dict(), 'model_ner/ner_link_round_%s.pth' % round)
    pred_vector.append(pred_set)
    round += 1
    # INFO = 'train loss %f, valid loss %f, acc %f, recall %f, f1 %f ' % (train_loss, valid_loss, INFO_THRE[0], INFO_THRE[1], INFO_THRE[2])
    # logging.info(INFO)
    # INFO = 'epoch %d, train loss %f, valid loss %f' % (epoch, train_loss, valid_loss)
    # logging.info(INFO + '\t' + INFO_THRE)
    #break
result = pd.DataFrame()
result['text'] = [x[0] for x in dev_X]
result['pos1'] = [x[1] for x in dev_X]
result['pos2'] = [x[2] for x in dev_X]
result['label'] = [x[3] for x in dev_X]
result['predict'] = pred_set

result.to_pickle('result/ner_link_part.pkl')
#pred_vector = np.concatenate(pred_vector, axis=0)
#np.save('result/gensim_vector_bert.npy', pred_vector)
    # 19 train loss　0.128424, val loss 0.153328, val_cos_loss 91155.319716
# 0.957            0.936170
# 0.945 0.923 0.945 0.943
# 0.932 0.941 0.95 0.943
#part 0 0.9/0.916/0.924/0.928
#part 1 0.953

# adam
# part0 0.921/0.9567/0.952/0.943 acc0.997
# part1 0.916/0.929/0.945/0.922
# rewrite metric
"""
2653 1653
100%|██████████| 213/213 [00:41<00:00,  5.43it/s]
阈值 [0.5049927]
  0%|          | 0/213 [00:00<?, ?it/s]real acc 0.993508,recall 0.975554,f1 0.984449
round 0 epoch 0 train loss　0.050168, val loss 0.048178  acc 0.976048, recall0.981928, f1 0.978979
100%|██████████| 213/213 [00:42<00:00,  5.44it/s]
阈值 [0.8562041]
real acc 0.997562,recall 0.978183,f1 0.987778
round 0 epoch 1 train loss　0.047379, val loss 0.047916  acc 0.990769, recall0.969880, f1 0.980213
100%|██████████| 213/213 [00:44<00:00,  4.98it/s]
阈值 [0.5467774]
real acc 0.997558,recall 0.976590,f1 0.986963
round 0 epoch 2 train loss　0.046908, val loss 0.047983  acc 0.990712, recall0.963855, f1 0.977099
100%|██████████| 213/213 [00:49<00:00,  4.98it/s]
阈值 [0.50223666]
real acc 0.997562,recall 0.978183,f1 0.987778
round 0 epoch 3 train loss　0.046647, val loss 0.047886  acc 0.990769, recall0.969880, f1 0.980213
"""
# default 五折 0.962 0.969 0.980 0.977 0.978
# add makeup*5 五折　0.985 0.980 0.982 0.987 0.985
# add makeup*1 五折 0.978 * 0.982 0.987
# nearest combination 0.998 0.996

