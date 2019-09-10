# coding:u8
import pandas as pd

dev_X = []
test_data = pd.read_pickle('result/ner_link.pkl')
print(test_data.shape)
for index, row in test_data.iterrows():
    s = row['text']
    for item in row['result']:
        print(item)
        a = item[0]
        b = item[1]
        dev_X.append((s, a, b, row['id']))
# print(test_data['result'])
# print(len(set(dev_X)))
for x in dev_X:
    print(x)
