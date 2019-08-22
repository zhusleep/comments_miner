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
from utils import seed_torch, read_data, load_glove, calc_f1, get_threshold
from pytorch_pretrained_bert import BertTokenizer,BertAdam
from sklearn.model_selection import KFold, train_test_split
import logging

from sklearn.externals import joblib

file_name = 'TRAIN/Train_reviews.csv'
file_labels = 'TRAIN/Train_labels.csv'
sentences = data_manager.read_nerlink(filename=file_name, filelabels=file_labels)
print(len(sentences),sum([x[3] for x in sentences]))
seed_torch(2019)
kfold = KFold(n_splits=5, shuffle=False, random_state=2019)
pred_vector = []
round = 0
for train_index, test_index in kfold.split(np.zeros(len(sentences))):
    # print(round)
    # if round<1:
    #     round +=1
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
    train_dataset = NERLINK([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in train_X], t,
                                pos1=[x[1] for x in train_X], pos2=[x[2] for x in train_X],
                                label=[x[3] for x in train_X], use_bert=True, gap=[x[4] for x in train_X])
    valid_dataset = NERLINK([['[CLS]'] + list(x[0]) + ['[CLS]'] for x in dev_X], t,
                                pos1=[x[1] for x in dev_X], pos2=[x[2] for x in dev_X],
                                label=[x[3] for x in dev_X], use_bert=True, gap=[x[4] for x in dev_X])
    train_batch_size = 10
    valid_batch_size = 10
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
            pos1 = pos1.type(torch.LongTensor).to(device)
            pos2 = pos2.type(torch.LongTensor).to(device)
            label = label.to(device).type(torch.float).view(-1,1)
            numerical_f = numerical_f.to(device)
            with torch.no_grad():
                pred = model(X, pos1, pos2, length, numerical_f)
            loss = loss_fn(pred, label)
            pred_set.append(pred.cpu().numpy())
            label_set.append(label.cpu().numpy())
            valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_X)
        pred_set = np.concatenate(pred_set, axis=0)
        label_set = np.concatenate(label_set, axis=0)
        # top_class = np.argmax(pred_set, axis=1)
        # equals = top_class == label_set
        # accuracy = np.mean(equals)
        # print('acc', accuracy)
        INFO_THRE, thre_list = get_threshold(pred_set, label_set)
        print('round', round, 'epoch', epoch,'train loss　%f, val loss %f ' % (train_loss, valid_loss), INFO_THRE)

    torch.save(model.state_dict(), 'model_ner/ner_link_round_%s.pth' % round)
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
real acc 0.990909,recall 0.955331,f1 0.972794
round 0 epoch 0 train loss　0.058550, val loss 0.054279  acc 0.965190, recall0.866477, f1 0.913174
100%|██████████| 283/283 [00:57<00:00,  5.08it/s]
阈值 [0.6068619]
real acc 0.987645,recall 0.955458,f1 0.971285
round 0 epoch 1 train loss　0.054317, val loss 0.053998  acc 0.953988, recall0.883523, f1 0.917404
100%|██████████| 283/283 [01:02<00:00,  4.84it/s]
阈值 [0.5062354]
real acc 0.983711,recall 0.962390,f1 0.972934
round 0 epoch 2 train loss　0.053659, val loss 0.053500  acc 0.942363, recall0.928977, f1 0.935622
100%|██████████| 283/283 [01:04<00:00,  4.24it/s]
阈值 [0.5831965]
real acc 0.990248,recall 0.970900,f1 0.980479
round 0 epoch 3 train loss　0.053159, val loss 0.053255  acc 0.964497, recall0.926136, f1 0.944928
100%|██████████| 283/283 [01:03<00:00,  4.53it/s]
阈值 [0.5311908]
  0%|          | 0/283 [00:00<?, ?it/s]real acc 0.984130,recall 0.938853,f1 0.960958
round 1 epoch 0 train loss　0.058787, val loss 0.055162  acc 0.939683, recall0.870588, f1 0.903817
100%|██████████| 283/283 [01:04<00:00,  4.78it/s]
阈值 [0.5054035]
real acc 0.984468,recall 0.959570,f1 0.971859
round 1 epoch 1 train loss　0.053797, val loss 0.054021  acc 0.944282, recall0.947059, f1 0.945668
100%|██████████| 283/283 [01:04<00:00,  4.69it/s]
阈值 [0.5358536]
  0%|          | 0/283 [00:00<?, ?it/s]real acc 0.990208,recall 0.966916,f1 0.978424
round 1 epoch 2 train loss　0.052658, val loss 0.053273  acc 0.963964, recall0.944118, f1 0.953938
100%|██████████| 283/283 [01:05<00:00,  4.78it/s]
阈值 [0.55167454]
real acc 0.991035,recall 0.968876,f1 0.979830
round 1 epoch 3 train loss　0.052076, val loss 0.053256  acc 0.966967, recall0.947059, f1 0.956909
100%|██████████| 283/283 [01:04<00:00,  4.89it/s]
阈值 [0.516608]
real acc 0.983386,recall 0.943267,f1 0.962909
round 2 epoch 0 train loss　0.058200, val loss 0.055016  acc 0.938080, recall0.878261, f1 0.907186
100%|██████████| 283/283 [01:04<00:00,  4.51it/s]
阈值 [0.4958054]
  0%|          | 0/283 [00:00<?, ?it/s]real acc 0.985291,recall 0.960733,f1 0.972857
round 2 epoch 1 train loss　0.054450, val loss 0.054036  acc 0.947059, recall0.933333, f1 0.940146
100%|██████████| 283/283 [01:04<00:00,  4.67it/s]
阈值 [0.52261376]
  0%|          | 0/283 [00:00<?, ?it/s]real acc 0.986203,recall 0.968271,f1 0.977155
round 2 epoch 2 train loss　0.053067, val loss 0.053375  acc 0.951009, recall0.956522, f1 0.953757
100%|██████████| 283/283 [01:04<00:00,  4.68it/s]
阈值 [0.57263356]
real acc 0.991093,recall 0.975251,f1 0.983108
round 2 epoch 3 train loss　0.052478, val loss 0.053031  acc 0.967742, recall0.956522, f1 0.962099
100%|██████████| 283/283 [01:04<00:00,  4.73it/s]
阈值 [0.60543007]
real acc 0.987417,recall 0.937928,f1 0.962037
round 3 epoch 0 train loss　0.057991, val loss 0.055442  acc 0.950658, recall0.873112, f1 0.910236
100%|██████████| 283/283 [01:04<00:00,  4.71it/s]
阈值 [0.49738282]
real acc 0.988426,recall 0.952637,f1 0.970202
round 3 epoch 1 train loss　0.053914, val loss 0.054387  acc 0.956250, recall0.924471, f1 0.940092
100%|██████████| 283/283 [01:04<00:00,  4.35it/s]
阈值 [0.5123158]
real acc 0.991725,recall 0.954900,f1 0.972964
round 3 epoch 2 train loss　0.052848, val loss 0.054274  acc 0.968051, recall0.915408, f1 0.940994
100%|██████████| 283/283 [01:04<00:00,  4.38it/s]
阈值 [0.51536024]
real acc 0.991759,recall 0.958884,f1 0.975045
round 3 epoch 3 train loss　0.052253, val loss 0.054048  acc 0.968553, recall0.930514, f1 0.949153
100%|██████████| 283/283 [01:04<00:00,  4.70it/s]
阈值 [0.6456892]
real acc 0.992610,recall 0.963235,f1 0.977702
round 4 epoch 0 train loss　0.057503, val loss 0.053785  acc 0.971963, recall0.888889, f1 0.928571
100%|██████████| 283/283 [01:04<00:00,  4.59it/s]
阈值 [0.67893314]
real acc 0.990287,recall 0.974884,f1 0.982525
round 4 epoch 1 train loss　0.053379, val loss 0.053016  acc 0.965015, recall0.943020, f1 0.953890
100%|██████████| 283/283 [01:04<00:00,  4.84it/s]
阈值 [0.5661336]
  0%|          | 0/283 [00:00<?, ?it/s]real acc 0.997576,recall 0.983761,f1 0.990620
round 4 epoch 2 train loss　0.052352, val loss 0.052467  acc 0.990964, recall0.937322, f1 0.963397
100%|██████████| 283/283 [01:04<00:00,  4.55it/s]
阈值 [0.4985642]
real acc 0.992724,recall 0.978375,f1 0.985497
round 4 epoch 3 train loss　0.052078, val loss 0.052527  acc 0.973529, recall0.943020, f1 0.958032
"""