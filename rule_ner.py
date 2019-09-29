import pandas as pd
import pickle

test_data = pd.read_pickle('result/ner_bert_result.pkl') #crf识别结果
test_data = test_data.reset_index().rename(columns={'index':'id'})
test_data['text'] = test_data['text'].apply(lambda x:''.join(x[1:-1]))
test_data['pos'] = test_data['pos'].apply(lambda x:[(item[0]-1,item[1]-1,item[2][1],item[2][2:]) for item in x])
test_xu = pickle.load(open('test_data_a_o_927.pkl', 'rb'))
rule_1 = [] # crf识别出实体类型为整体但是双指针没有识别出来的实体仍然放进最终结果
rule_2 = [] # crf识别出的单实体但是双指针没有识别出来

for index,item in enumerate(test_xu):
    temp = []
    if 'a' in item:
        for a_part in item['a']:
            temp.append((a_part[1], a_part[2]-1, 'A', a_part[5]))
    if 'o' in item:
        for o_part in item['o']:
            temp.append((o_part[1], o_part[2]-1, 'O', o_part[5]))
    temp = sorted(temp, key=lambda x: x[0], reverse=False)
    row = test_data.loc[index,:]
    for a in row['pos']:
        # crf识别出实体类型为整体但是双指针没有识别出来的实体仍然放进最终结果
        if a not in temp and a[3]=='整体' and '方便' not in item['text'][a[0]:a[1]+1] and len(temp)==0:
            conflict = False
            for b in temp:
                if a[0]==b[0] or a[1]==b[1]:
                    conflict =True
                    break
            if not conflict:
                rule_1.append((row['id'],a))
                print(row['id'],row['text'])
                print(row['pos'],'\n',temp)
                # 加入item中
                new_ner = (item['text'][a[0]:a[1]+1],a[0],a[1]+1,0.95,0.95,a[3])
                if a[2]=='A':
                    if 'a' in item:
                        item['a'].append(new_ner)
                    else:
                        item['a'] = [new_ner]
                if a[2]=='O':
                    if 'o' in item:
                        item['o'].append(new_ner)
                    else:
                        item['o'] = [new_ner]
        # crf识别出的单实体但是双指针没有识别出来
        if a not in temp and a[1]==a[0] and len(temp)==0:# and '方便' not in row['text'][a[0]:a[1]+1]:
            conflict = False
            for b in temp:
                if a[0]==b[0] or a[1]==b[1]:
                    conflict =True
                    break
            if not conflict and (row['id'],a) not in rule_1 :
                rule_2.append((row['id'],a))
                print(row['id'],row['text'])
                print(row['pos'],'\n',temp)
                new_ner = (item['text'][a[0]:a[1]+1],a[0],a[1]+1,0.95,0.95,a[3])
                if a[2]=='A':
                    if 'a' in item:
                        item['a'].append(new_ner)
                    else:
                        item['a'] = [new_ner]
                if a[2]=='O':
                    if 'o' in item:
                        item['o'].append(new_ner)
                    else:
                        item['o'] = [new_ner]

print(len(rule_1), len(rule_2))
pickle.dump(test_xu, open('test_data_a_o_927_rule.pkl', 'wb'))


