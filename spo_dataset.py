#-*-coding:utf-8-*-
from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm as tqdm


class SPO(Dataset):
    def __init__(self, X, tokenizer, max_len=50, label=None,  ner=None, combined_char_t=None):
        super(SPO, self).__init__()
        self.max_len = max_len
        X = self.limit_length(X)
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.X = tokenizer.transform(X)
        # X = pad_sequences(X, maxlen=198)
        self.length = [len(sen) for sen in self.X]
        # self.numerical_df = self.cal_numerical_f(self.X)
        self.ner = ner
        self.combined_char_t = combined_char_t

    def limit_length(self, X):
        temp = []
        for item in X:
            temp.append(item[0:self.max_len])
        return temp

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        ner = torch.tensor(self.ner[index])
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if ner is not None:
            return index, sentence, ner,length
        else:
            return index, sentence,length


class QAPair(Dataset):
    def __init__(self, data, tokenizer, predictor, max_len=50):
        super(QAPair, self).__init__()

        self.data = data
        self.question = self.data['question'].apply(lambda x: tokenizer.transform([x])[0]).reset_index(drop=True)
        self.answer = self.data['answer'].apply(lambda x: tokenizer.transform([x])[0]).reset_index(drop=True)
        self.la = self.data['la'].reset_index(drop=True)
        self.lb = self.data['lb'].reset_index(drop=True)
        self.label = self.data['label'].reset_index(drop=True)
        self.numerical_df = self.data[predictor].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #print('index', index)
        sa = torch.tensor(self.question[index], dtype=torch.int)
        sb = torch.tensor(self.answer[index], dtype=torch.int)
        la = self.la[index]
        lb = self.lb[index]
        numerical_features = self.numerical_df.loc[index, :].values

        label = self.label[index]
        return sa, sb, la, lb, numerical_features, label


def qapair_collate_fn(batch):
    if (len(batch[0]) == 6):
        sa, sb, la, lb, numerical_feats, label = zip(*batch)
        la = torch.tensor(la)
        lb = torch.tensor(lb)
        numerical_feats = torch.tensor(numerical_feats)
        label = torch.tensor(label)
        return sa, sb, la, lb, numerical_feats, label
    else:
        has_label = len(batch[0]) == 8

        if has_label:
            sa, sb, la, lb, cata, catb, numerical_feats, label = zip(*batch)
            label = torch.tensor(label)
        else:
            sa, sb, la, lb, cata, catb, numerical_feats = zip(*batch)

        la = torch.tensor(la)
        lb = torch.tensor(lb)
        cata = torch.tensor(cata)
        catb = torch.tensor(catb)
        numerical_feats = torch.tensor(numerical_feats)

        if has_label:
            return sa, sb, la, lb, cata, catb, numerical_feats, label
        else:
            return sa, sb, la, lb, cata, catb, numerical_feats



class SPO_BERT(Dataset):
    def __init__(self, X, tokenizer, label=None,  ner=None):
        super(SPO_BERT, self).__init__()
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.X = self.deal_for_bert(X, self.tokenizer)
        # X = pad_sequences(X, maxlen=198)
        self.ner = ner
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self,x,t):
        # text = {}
        # for s in x:
        #     for word in s:
        #         if word in text:
        #             text[word] += 1
        #         else:
        #             text[word] = 1
        # extra_token_id = 1
        # for w in sorted(text.items(), key=lambda x: x[1])[-90:]:
        #     self.tokenizer.vocab[w[0]] = extra_token_id
        #     extra_token_id += 1

        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            #sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        if self.ner is not None:
            ner1 = torch.tensor(self.ner[index][0])
            ner2 = torch.tensor(self.ner[index][1])
            ner3 = torch.tensor(self.ner[index][2])
            ner4 = torch.tensor(self.ner[index][3])
            ner5 = torch.tensor(self.ner[index][4])

        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if  self.ner is not None:
            return index, sentence, ner1, ner2, ner3, ner4, ner5, length
        else:
            return index, sentence, length


class SPO_BERT_nerornot(Dataset):
    def __init__(self, X, tokenizer, label=None,  ner=None):
        super(SPO_BERT_nerornot, self).__init__()
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.X = self.deal_for_bert(X, self.tokenizer)
        # X = pad_sequences(X, maxlen=198)
        self.ner = ner
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self,x,t):
        # text = {}
        # for s in x:
        #     for word in s:
        #         if word in text:
        #             text[word] += 1
        #         else:
        #             text[word] = 1
        # extra_token_id = 1
        # for w in sorted(text.items(), key=lambda x: x[1])[-90:]:
        #     self.tokenizer.vocab[w[0]] = extra_token_id
        #     extra_token_id += 1

        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            #sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        if self.ner is not None:
            ner = torch.tensor(self.ner[index])

        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if  self.ner is not None:
            return index, sentence, ner, length
        else:
            return index, sentence, length



class SPO_BERT_LINK(Dataset):
    def __init__(self, X, tokenizer, pos, type=None, max_len=510):
        super(SPO_BERT_LINK, self).__init__()
        self.raw_X = X
        self.type = type
        self.tokenizer = tokenizer
        self.X = self.deal_for_bert(X, self.tokenizer)
        # X = pad_sequences(X, maxlen=198)
        self.pos = pos
        self.max_len = max_len
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self,x,t):
        bert_tokens = []
        for item in x:
            item = item[0:510]
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            #sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        pos = self.pos[index]
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:

        if self.type is not None:
            type = self.type[index]
            return index, sentence, type, pos, length
        else:
            return index, sentence, pos, length


class entity_linking_v2(Dataset):
    def __init__(self, X, tokenizer, max_len=510):
        super(entity_linking_v2, self).__init__()
        # self.raw_X = X
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.X = self.deal_for_bert(X)
        # X = pad_sequences(X, maxlen=198)
        # self.length = [len(sen) for sen in self.X]

    def bert_tokenize(self,x):
        temp = []
        for w in x:
            if w in self.tokenizer.vocab:
                temp.append(w)
            else:
                temp.append('[UNK]')
        temp = temp[0:self.max_len]
        temp = ['[CLS]'] + temp + ['[SEP]']
        # sen = t.tokenize(''.join(item))
        bert_tokens = self.tokenizer.convert_tokens_to_ids(temp)
        return bert_tokens

    def deal_for_bert(self, X):
        new_x = []
        print('deal for bert tokenization')
        for sen in tqdm(X):
            sen['query'] = self.bert_tokenize(sen['query'])
            sen['candidate_abstract'] = self.bert_tokenize(sen['candidate_abstract'])
            sen['pos'] = [sen['pos'][0]+1, sen['pos'][1]+1]
            new_x.append(sen)
        return new_x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.X[index]['label']
        query = torch.tensor(self.X[index]['query'])
        pos = self.X[index]['pos']
        candidate_abstract = torch.tensor(self.X[index]['candidate_abstract'])
        candidate_type = self.X[index]['candidate_type']

        # l_abstract = len(candidate_abstract)
        # l_query = len(query)
        # l_labels = len(candidate_labels)

        if label is not None:
            return index, label, query, pos, candidate_abstract, candidate_type
        # if label is not None:
        #     return index, label, query, l_query, pos, candidate_abstract, l_abstract, candidate_labels, l_labels, \
        #            candidate_type, candidate_abstract_numwords, candidate_numattrs
        # else:
        #     return index


class deep_distance_dataset(Dataset):
    def __init__(self, X, tokenizer, max_len=510):
        super(deep_distance_dataset, self).__init__()
        # self.raw_X = X
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.X = self.deal_for_bert(X)
        # X = pad_sequences(X, maxlen=198)
        # self.length = [len(sen) for sen in self.X]

    def bert_tokenize(self,x):
        temp = []
        for w in x:
            if w in self.tokenizer.vocab:
                temp.append(w)
            else:
                temp.append('[UNK]')
        temp = temp[0:self.max_len]
        temp = ['[CLS]'] + temp + ['[SEP]']
        # sen = t.tokenize(''.join(item))
        bert_tokens = self.tokenizer.convert_tokens_to_ids(temp)
        return bert_tokens

    def deal_for_bert(self, X):
        new_x = []
        print('deal for bert tokenization')
        for sen in tqdm(X):
            sen['query'] = self.bert_tokenize(sen['query'])
            sen['candidate_abstract'] = self.bert_tokenize(sen['candidate_abstract'])
            sen['pos'] = [sen['pos'][0]+1, sen['pos'][1]+1]
            new_x.append(sen)
        return new_x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        label = self.X[index]['label']
        query = torch.tensor(self.X[index]['query'])
        pos = self.X[index]['pos']
        candidate_abstract = torch.tensor(self.X[index]['candidate_abstract'])
        pos_answer = self.X[index]['pos_answer']
        # l_abstract = len(candidate_abstract)
        # l_query = len(query)
        # l_labels = len(candidate_labels)

        if label is not None:
            return index, label, query, pos, candidate_abstract, pos_answer

class entity_linking_v3(Dataset):
    def __init__(self, X, tokenizer, max_len=500):
        super(entity_linking_v3, self).__init__()
        # self.raw_X = X
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.X = self.deal_for_tokenization(X)
        # X = pad_sequences(X, maxlen=198)
        # self.length = [len(sen) for sen in self.X]

    def deal_for_tokenization(self, X):
        text = []
        for one_batch in X:
            for sen in one_batch:
                text.append(sen['query'])
                text.append(sen['candidate_abstract'])
                text.append(sen['candidate_labels'])
        self.tokenizer.fit(text)

        new_x = []
        print('deal for tokenization')
        for one_batch in tqdm(X):
            batch_sample = []
            for sen in one_batch:
                sen['query'] = self.tokenizer.transform([sen['query']])[0]
                sen['candidate_abstract'] = self.tokenizer.transform([sen['candidate_abstract']])[0]
                sen['candidate_labels'] = self.tokenizer.transform([sen['candidate_labels']])[0]
                sen['pos'] = [sen['pos'][0], sen['pos'][1]]
                if not sen['candidate_abstract']:
                    sen['candidate_abstract'] = [1]
                if not sen['candidate_labels']:
                    sen['candidate_labels'] = [1]
                batch_sample.append(sen)
            new_x.append(batch_sample)
        return new_x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index]

        label = self.X[index]['label']
        query = torch.tensor(self.X[index]['query'])
        pos = self.X[index]['pos']
        candidate_abstract = torch.tensor(self.X[index]['candidate_abstract'])
        candidate_labels = torch.tensor(self.X[index]['candidate_labels'])
        candidate_type = self.X[index]['candidate_type']
        candidate_abstract_numwords = self.X[index]['candidate_abstract_numwords']
        candidate_numattrs = self.X[index]['candidate_numattrs']

        l_abstract = len(candidate_abstract)
        l_query = len(query)
        l_labels = len(candidate_labels)

        if label is not None:
            return index, label, query, l_query, pos, candidate_abstract, l_abstract, candidate_labels, l_labels, \
                   candidate_type, candidate_abstract_numwords, candidate_numattrs
        else:
            return index


class SPO_LINK(Dataset):
    def __init__(self, X, tokenizer, pos, type=None):
        super(SPO_LINK, self).__init__()
        self.raw_X = X
        self.type = type
        self.tokenizer = tokenizer
        self.X = tokenizer.transform(X)        # X = pad_sequences(X, maxlen=198)
        self.pos = pos
        self.length = [len(sen) for sen in self.X]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        pos = self.pos[index]
        type = self.type[index]
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        #if self.combined_char_t is not None:
        if pos[1]>length:
            raise Exception
        if type is not None:
            return index, sentence, type, pos, length
        else:
            return index, sentence, length


class NERLINK(Dataset):
    def __init__(self, X, tokenizer, pos1, pos2, label=None, use_bert=False,gap =None):
        super(NERLINK, self).__init__()
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        self.gap = gap
        if not use_bert:
            self.X = tokenizer.transform(X)        # X = pad_sequences(X, maxlen=198)
        else:
            self.X = self.deal_for_bert(X, self.tokenizer, pos1, pos2)
        self.pos1 = pos1
        self.pos2 = pos2
        self.type_error = [1 if a[3]==b[3] else 0 for a,b in zip(self.pos1,self.pos2)]
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self, x, t, pos1, pos2):
        bert_tokens = []
        for index, item in enumerate(x):
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')
            # 在ner周围添加unused标记
            temp.insert(pos1[index][0]+1, '[unused1]')
            temp.insert(pos1[index][1]+3, '[unused2]')
            temp.insert(pos2[index][0]+3, '[unused3]')
            temp.insert(pos2[index][1]+5, '[unused4]')
            # sen = t.tokenize(''.join(item))
            # 截取中间片段
            #temp = [temp[0]]+temp[pos1[index][0]+1:pos2[index][1]+2]+[temp[-1]]
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def numerical_f(self, sentence, pos1, pos2, index):
        feature = []
        # A first
        if pos1[2] == 'A':
            feature.append(1)
        else:
            feature.append(0)
        # continue
        if pos1[1]+1==pos2[0]:
            feature.append(1)
        else:
            feature.append(0)
        # pos gap
        feature.append(pos2[0]-pos1[1])
        # splitter
        split = set(['，', '！', ' ', '?', '。', '\n', '～'])
        count_split = 0
        for item in sentence[pos1[1]: pos2[0]]:
            if item in split:
                count_split += 1
        feature.append(count_split)
        feature.append(pos1[1]-pos1[0])
        feature.append(pos2[1]-pos2[0])
        # index gap
        #feature.append(1)
        feature.append(self.gap[index])
        if pos2[3]==pos1[3]:
            feature.append(1)
        else:
            feature.append(0)
        return feature

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        numerical_f = self.numerical_f(self.raw_X[index], self.pos1[index], self.pos2[index], index)
        pos1 = [self.pos1[index][0]+2, self.pos1[index][1]+2]
        pos2 = [self.pos2[index][0]+4, self.pos2[index][1]+4]

        if self.label is not None:
            label = self.label[index]
        else:
            label = None
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        # if self.combined_char_t is not None:
        if pos2[1] > length:
            raise Exception
        if label is not None:
            return index, sentence, label, pos1, pos2, length, numerical_f
        else:
            return index, sentence, pos1, pos2, length, numerical_f


class NERCate(Dataset):
    def __init__(self, X, tokenizer, pos1, pos2, label=None, use_bert=False):
        super(NERCate, self).__init__()
        self.raw_X = X
        self.label = label
        self.tokenizer = tokenizer
        if not use_bert:
            self.X = tokenizer.transform(X)        # X = pad_sequences(X, maxlen=198)
        else:
            self.X = self.deal_for_bert(X, self.tokenizer)
        self.pos1 = pos1
        self.pos2 = pos2
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self, x, t):
        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            # sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        numerical_f = [0]
        c = True
        if not c:
            pos1 = [0, 0]

            try:
                pos2 = [self.pos2[index][0] + 1, self.pos2[index][1] + 1]
            except:
                pos2 = [self.pos1[index][0] + 1, self.pos1[index][1] + 1]
        else:
            try:
                pos1 = [self.pos1[index][0]+1, self.pos1[index][1]+1]
            except:
                pos1 = [0, 0]

            try:
                pos2 = [self.pos2[index][0]+1, self.pos2[index][1]+1]
            except:
                pos2 = [0, 0]

        if self.label is not None:
            label = self.label[index]
        else:
            label = None
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        # if self.combined_char_t is not None:
        if pos2[1] > length:
            raise Exception
        if label is not None:
            return index, sentence, label, pos1, pos2, length, numerical_f
        else:
            return index, sentence, pos1, pos2, length, numerical_f


class NERCateMT(Dataset):
    def __init__(self, X, tokenizer, pos1, pos2, polar_label=None,cate_label=None, use_bert=False):
        super(NERCateMT, self).__init__()
        self.raw_X = X
        self.cate_label = cate_label
        self.polar_label=polar_label
        self.tokenizer = tokenizer
        if not use_bert:
            self.X = tokenizer.transform(X)        # X = pad_sequences(X, maxlen=198)
        else:
            self.X = self.deal_for_bert(X, self.tokenizer)
        self.pos1 = pos1
        self.pos2 = pos2
        self.length = [len(sen) for sen in self.X]

    def deal_for_bert(self, x, t):
        bert_tokens = []
        for item in x:
            temp = []
            for w in item:
                if w in self.tokenizer.vocab:
                    temp.append(w)
                else:
                    temp.append('[UNK]')

            # sen = t.tokenize(''.join(item))
            indexed_tokens = t.convert_tokens_to_ids(temp)
            bert_tokens.append(indexed_tokens)
        return bert_tokens

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sentence = torch.tensor(self.X[index])
        numerical_f = [0]
        c = True
        if not c:
            pos1 = [0, 0]

            try:
                pos2 = [self.pos2[index][0] + 1, self.pos2[index][1] + 1]
            except:
                pos2 = [self.pos1[index][0] + 1, self.pos1[index][1] + 1]
        else:
            try:
                pos1 = [self.pos1[index][0]+1, self.pos1[index][1]+1]
            except:
                pos1 = [0, 0]

            try:
                pos2 = [self.pos2[index][0]+1, self.pos2[index][1]+1]
            except:
                pos2 = [0, 0]

        if self.polar_label is not None:
            polar_label = self.polar_label[index]
            cate_label = self.cate_label[index]

        else:
            polar_label = None
            cate_label = None
        length = self.length[index]
        # if self.ner is not None:
        #     ner = torch.tensor(self.ner[index])
        # if self.combined_char_t is not None:
        if pos2[1] > length:
            raise Exception
        if polar_label is not None:
            return index, sentence, cate_label, polar_label, pos1, pos2, length, numerical_f
        else:
            return index, sentence, pos1, pos2, length, numerical_f


def pad_sequence(sequences):
    max_len = max([len(s) for s in sequences])
    out_mat = [[torch.tensor([0])]*max_len]*len(sequences)
    for i, sen in enumerate(sequences):
        sen = [torch.tensor(w) for w in sen]
        length = len(sen)
        out_mat[i][:length] = sen

    return out_mat


def collate_fn(batch):

    if len(batch[0]) == 8:
        index, X, ner1,ner2,ner3,ner4,ner5, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        return index, X, ner1,ner2,ner3,ner4,ner5, length,
    else:
        index, X, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        return index, X, length,


def collate_fn_nerornot(batch):

    if len(batch[0]) == 4:
        index, X, ner1, length = zip(*batch)
        ner1 = torch.tensor(ner1)

        length = torch.tensor(length, dtype=torch.int)
        return index, X, ner1, length,
    else:
        index, X, length = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        return index, X, length,


def collate_fn_ner_link(batch):

    if len(batch[0]) == 7:
        index, sentence, label, pos1, pos2, length, numerical_f = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        numerical_f = torch.tensor(numerical_f, dtype=torch.float)
        length = torch.tensor(length, dtype=torch.int)
        label = torch.tensor(label, dtype=torch.long)
        return index, sentence, label, pos1, pos2, length, numerical_f
    else:
        index, sentence, pos1, pos2, length, numerical_f = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        numerical_f = torch.tensor(numerical_f, dtype=torch.float)
        length = torch.tensor(length, dtype=torch.int)
        return index, sentence, pos1, pos2, length, numerical_f


def collate_fn_ner_cate(batch):

    if len(batch[0]) == 7:
        index, sentence, label, pos1, pos2, length, numerical_f = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        length = torch.tensor(length, dtype=torch.int)
        label = torch.tensor(label, dtype=torch.long)
        numerical_f = torch.tensor(numerical_f, dtype=torch.float)
        return index, sentence, label, pos1, pos2, length, numerical_f
    else:
        index, sentence, pos1, pos2, length, numerical_f = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        length = torch.tensor(length, dtype=torch.int)
        numerical_f = torch.tensor(numerical_f, dtype=torch.float)
        return index, sentence, pos1, pos2, length, numerical_f


def collate_fn_ner_cate_mt(batch):

    if len(batch[0]) == 8:
        index, sentence, cate_label, polar_label, pos1, pos2, length, numerical_f = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        length = torch.tensor(length, dtype=torch.int)
        cate_label = torch.tensor(cate_label, dtype=torch.long)
        polar_label = torch.tensor(polar_label, dtype=torch.long)
        numerical_f = torch.tensor(numerical_f, dtype=torch.float)
        return index, sentence, cate_label,polar_label, pos1, pos2, length, numerical_f
    else:
        index, sentence, pos1, pos2, length = zip(*batch)
        pos1 = torch.tensor(pos1, dtype=torch.int)
        pos2 = torch.tensor(pos2, dtype=torch.int)
        length = torch.tensor(length, dtype=torch.int)
        return index, sentence, pos1, pos2, length


def collate_fn_linking_v2(batch):

    if len(batch[0]) == 6:
        index, label, query, pos, candidate_abstract, candidate_type = zip(*batch)

        pos = torch.tensor(pos, dtype=torch.int)
        candidate_type = torch.tensor(candidate_type, dtype=torch.long)
        label = torch.tensor(label)
        # candidate_abstract_numwords = torch.tensor(candidate_abstract_numwords)
        # candidate_numattrs = torch.tensor(candidate_numattrs)
        return index, label, query, pos, candidate_abstract, candidate_type
        # index, X, length = zip(*batch)
        # length = torch.tensor(length, dtype=torch.long)
        # return index, X, length,


def collate_fn_linking_deep_distance(batch):

    if len(batch[0]) == 6:
        index, label, query, pos, candidate_abstract, pos_answer = zip(*batch)

        pos = torch.tensor(pos, dtype=torch.int)
        pos_answer = torch.tensor(pos_answer, dtype=torch.int)
        label = torch.tensor(label)
        # candidate_abstract_numwords = torch.tensor(candidate_abstract_numwords)
        # candidate_numattrs = torch.tensor(candidate_numattrs)
        return index, label, query, pos, candidate_abstract, pos_answer



def collate_fn_ner(batch):

    if len(batch[0]) == 7:
        index, X, pos, length, numerical_feats, ner, label = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        label = torch.tensor(label)
        return index, X, pos, length, numerical_feats, ner, label
    else:
        index, X, pos, length, numerical_feats = zip(*batch)
        length = torch.tensor(length, dtype=torch.int)
        numerical_feats = torch.tensor(numerical_feats)
        return index, X, pos, length, numerical_feats


def get_mask(sequences_batch, sequences_lengths, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.uint8)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    if is_cuda:
        return mask.cuda()
    else:
        return mask


def get_mask_attn_pool(sequences_batch, sequences_lengths, pos1, pos2, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.uint8)
    for i in range(batch_size):
        if pos1[i][0] !=0:
            mask[i,pos1[i][0]:pos1[i][1]+1] = 1
        if pos2[i][0] !=0:
            mask[i,pos2[i][0]:pos2[i][1]+1] = 1
    #mask[sequences_batch[:, :max_length] == 0] = 0.0
    if is_cuda:
        return mask.cuda()
    else:
        return mask


def get_mask_pos(sequences_batch, sequences_lengths, pos1, pos2, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask1 = torch.zeros(batch_size, max_length, dtype=torch.uint8)
    mask2 = torch.zeros(batch_size, max_length, dtype=torch.uint8)

    for i in range(batch_size):
        mask1[i,pos1[i][0]:pos1[i][1]+1] = 1
        mask2[i,pos2[i][0]:pos2[i][1]+1] = 1
    return mask1,mask2
    #mask[sequences_batch[:, :max_length] == 0] = 0.0


def get_mask_bertpiece(sequences_batch, sequences_lengths, pos, is_cuda=True):
    """
    Get the mask for a batch of padded variable length sequences.
    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).
    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.zeros(batch_size, max_length, dtype=torch.uint8)
    for i in range(batch_size):
        mask[i, pos[i][0]:pos[i][1]+1] = 1
    if is_cuda:
        return mask.cuda()
    else:
        return mask

