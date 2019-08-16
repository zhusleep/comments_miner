#-*-coding:utf-8-*-
import pandas as pd
import numpy as np


class DataManager(object):
    def __init__(self):
        self.ner_list = ['O','B-A','I-A','B-O','I-O']
        np.random.seed(2019)

    def BIO(self, m, type):
        width = m[1]-m[0]
        if width == 1:
            result = ['B-'+type]
        else:
            result = ['I-'+type] * width
            result[0] = 'B-'+type
        #print(result)
        return [self.ner_list.index(x) for x in result]

    def read_basic_info(self,filename,filelabels):
        train = pd.read_csv(filename)
        labels = pd.read_csv(filelabels)
        # 标注错误
        labels.loc[6222, 'O_start'] = 14
        labels.loc[6222, 'O_end'] = 16
        labels.loc[5613, 'O_start'] = 30
        labels.loc[5613, 'O_end'] = 32
        labels.loc[2488, 'O_start'] = 28
        labels.loc[2488, 'O_end'] = 30
        labels.loc[2224, 'O_start'] = 18
        labels.loc[2224, 'O_end'] = 20
        labels.loc[2610, 'O_start'] = 44
        labels.loc[2610, 'O_end'] = 46
        labels.loc[2146, 'O_start'] = 15
        labels.loc[2146, 'O_end'] = 17
        labels.loc[1858, 'O_start'] = 38
        labels.loc[1858, 'O_end'] = 39
        labels.loc[866, 'A_start'] = 26
        labels.loc[866, 'A_end'] = 28
        labels.loc[2488, 'O_start'] = 5
        labels.loc[2488, 'O_end'] = 7
        labels.loc[3440, 'AspectTerms'] = '_'
        labels.loc[3440, 'Categories'] = '整体'
        labels.loc[316, 'AspectTerms'] = '_'

        # aspect and opinion是否连续
        span = {}
        for index, row in labels.iterrows():
            if row['id'] not in span:
                span[row['id']] = []
            if row['AspectTerms'] != '_':
                span[row['id']].append((int(row['A_start']), int(row['A_end']), 'A'))
            if row['OpinionTerms'] != '_':
                span[row['id']].append((int(row['O_start']), int(row['O_end']), 'O'))
        for id in span:
            span[id] = sorted(list(set(span[id])), key=lambda x: x[0], reverse=False)
        self.span = span
        self.train = train
        self.labels = labels

    def parseData(self, filename, filelabels):
        self.read_basic_info(filename,filelabels)

        sentence = []
        ner_labels = []
        for index,row in self.train.iterrows():
            sentence.append(row['Reviews'])
            ner = np.array([0] * len(row['Reviews']))
            for mention_ner in self.span[row['id']]:
                ner[mention_ner[0]:mention_ner[1]] = self.BIO(mention_ner[0:2], mention_ner[-1])
            ner_labels.append(ner)
        assert len(ner_labels)==len(sentence)

        return sentence,ner_labels

data_manager = DataManager()