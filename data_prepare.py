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

    def read_raw_data(self,filename,filelabels):
        train = pd.read_csv(filename)
        labels = pd.read_csv(filelabels)
        # 标注错误
        # 标注错误
        # 标注错误
        # 纠错逻辑　标注人员将多个很好标注为一个很好，位置错位
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
        labels.loc[1931, 'A_end'] = 11
        labels.loc[1931, 'O_end'] = 14

        labels.loc[1699, 'Polarities'] = '负面'
        labels.loc[1700, 'Polarities'] = '负面'
        labels.loc[1701, 'Polarities'] = '负面'
        labels.loc[1679, 'Polarities'] = '负面'
        labels.loc[1977, 'Polarities'] = '负面'
        labels.loc[2304, 'Polarities'] = '负面'

        labels.loc[3440, 'AspectTerms'] = '_'
        labels.loc[3440, 'Categories'] = '整体'
        labels.loc[1446, 'Categories'] = '整体'
        labels.loc[1684, 'Categories'] = '整体'
        labels.loc[1748, 'Categories'] = '包装'
        labels.loc[2228, 'Categories'] = '尺寸'
        # 纠错逻辑　标注人员对性价比的标注应当是价格
        labels.loc[1305, 'Categories'] = '价格'
        labels.loc[5135, 'Categories'] = '价格'
        labels.loc[2853, 'Categories'] = '价格'
        labels.loc[4298, 'Categories'] = '功效'

        labels.loc[316, 'AspectTerms'] = '_'
        train['Reviews'] = train['Reviews'].apply(lambda x: x.replace('增品', '赠品'))
        # 纠错逻辑　labels[(labels.AspectTerms!='_')&(labels.Categories=='整体')]　目标标注人员对于整体这个类别的aspect都是不标的
        labels.loc[858, 'Categories'] = '气味'
        labels.loc[1392, 'AspectTerms'] = '_'
        labels.loc[1392, 'OpinionTerms'] = '好评'
        labels.loc[1392, 'O_start'] = 41
        labels.loc[1392, 'O_end'] = 43
        labels.loc[1906, 'Categories'] = '包装'
        labels.loc[2008, 'Categories'] = '其他'
        labels.loc[2689, 'Categories'] = '功效'
        labels.loc[3460, 'Categories'] = '功效'
        labels.loc[3890, 'Categories'] = '速度'

        return train, labels

    def read_ner_basic_info(self,filename, filelabels):
        train, labels = self.read_raw_data(filename,filelabels)
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
        self.read_ner_basic_info(filename, filelabels)
        sentence = []
        ner_labels = []
        for index, row in self.train.iterrows():
            sentence.append(row['Reviews'])
            ner = np.array([0] * len(row['Reviews']))
            for mention_ner in self.span[row['id']]:
                ner[mention_ner[0]:mention_ner[1]] = self.BIO(mention_ner[0:2], mention_ner[-1])
            ner_labels.append(ner)
        assert len(ner_labels) == len(sentence)
        return sentence, ner_labels

    def read_nerlink(self, filename, filelabels):
        train, labels = self.read_raw_data(filename, filelabels)
        # aspect and opinion是否连续
        span = {}
        AO_connect = []
        for index, row in labels.iterrows():
            if row['id'] not in span:
                span[row['id']] = []
            if row['AspectTerms'] != '_':
                span[row['id']].append((int(row['A_start']), int(row['A_end'])-1, 'A'))
            if row['OpinionTerms'] != '_':
                span[row['id']].append((int(row['O_start']), int(row['O_end'])-1, 'O'))
            if row['AspectTerms'] != '_' and row['OpinionTerms'] != '_':
                [a,b] = sorted([(int(row['A_start']), int(row['A_end'])-1, 'A'), (int(row['O_start']), int(row['O_end'])-1,'O')],key=lambda x:x[0],reverse=False)
                AO_connect.append((row['id'],a,b))
        AO_connect = set(AO_connect)
        for id in span:
            span[id] = sorted(list(set(span[id])), key=lambda x: x[0], reverse=False)

        idToText = train['Reviews'].to_dict()
        # find potential A-O connection
        training = []
        for id in span:
            for index, item in enumerate(span[id]):
                if item[-1] == 'A':
                    for other_index in [index-1,index+1]:
                        if other_index>=0 and other_index<=len(span[id])-1 and span[id][other_index][-1] != 'A':
                            [a,b] = sorted([index,other_index])
                            sen = idToText[id-1]
                            if (id,span[id][a],span[id][b]) in AO_connect:
                                training.append((sen,span[id][a],span[id][b],1,abs(other_index-index)-1))
                            else:
                                training.append((sen,span[id][a],span[id][b],0,abs(other_index-index)-1))
        return training

    def read_ner_cate(self, filename, filelabels):
        self.read_ner_basic_info(filename, filelabels)
        self.category = list(self.labels['Categories'].unique())
        self.polarity = list(self.labels['Polarities'].unique())
        sentences = []
        reviews = self.train['Reviews'].to_dict()
        for index, row in self.labels.iterrows():
            if row['AspectTerms'] != '_' and row['OpinionTerms'] != '_':
                s = reviews[row['id']-1]
                a = (int(row['A_start']), int(row['A_end'])-1)
                b = (int(row['O_start']), int(row['O_end'])-1)
                sentences.append((s,a,b,self.category.index(row['Categories']),self.polarity.index(row['Polarities'])))
            if row['AspectTerms'] != '_' and row['OpinionTerms'] == '_':
                s = reviews[row['id']-1]
                a = (int(row['A_start']), int(row['A_end'])-1)
                sentences.append((s,a,None,self.category.index(row['Categories']),self.polarity.index(row['Polarities'])))
            if row['AspectTerms'] == '_' and row['OpinionTerms'] != '_':
                s = reviews[row['id']-1]
                b = (int(row['O_start']), int(row['O_end'])-1)
                sentences.append((s,None,b,self.category.index(row['Categories']),self.polarity.index(row['Polarities'])))
        return sentences


data_manager = DataManager()


