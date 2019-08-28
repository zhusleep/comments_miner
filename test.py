
from data_prepare import data_manager
from utils import seed_torch, read_data, load_glove, calc_f1, cal_ner_result
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
label_result = calc_f1(labels, labels, data_manager)
print(label_result)