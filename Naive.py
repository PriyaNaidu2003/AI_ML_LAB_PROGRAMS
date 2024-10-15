import csv
import math
import random
import statistics

def cal_prob(x,mean,stdev):
    expo=math.exp(-(math.pow(x-mean, 2)/(2*math.pow(stdev, 2))))
    return (1/(math.sqrt(2*math.pi)*stdev))*expo
dataset=[]
dataset_size=0
with open('navie.csv') as csvfile:
    lines=csv.reader(csvfile)
    for r in lines:
        dataset.append([float(att) for att in r])
dataset_size=len(dataset)
print("size of data set is : ",dataset_size)
train_size=int(0.7*dataset_size)
print(train_size)
x_train=[]
x_test=dataset.copy()
training_index=random.sample(range(dataset_size),train_size)
for i in training_index:
    x_train.append(dataset[i])
    x_test.append(dataset[i])
classes={}
for samp in x_train:
    last=int(samp[-1])
    if last not in classes:
        classes[last]=[]
    classes[last].append(samp)
print(classes)
summaries={}
for classval,train_data in classes.items():
    summary=[(statistics.mean(att),statistics.stdev(att)) for att in zip(*train_data)]
    del summary[-1]
    summaries[classval]=summary
print(summaries)
x_pred=[]
for i in x_test:
    probabilities={}
    for classval,classsummary in summaries.items():
        probabilities[classval]=1
        for index,alt in enumerate(classsummary):
            probabilities[classval]=cal_prob(i[index], alt[0], alt[1])
        best_label,best_prob=None,-1
        for classval,probability in probabilities.items():
            if best_label is None or probability>best_prob:
                best_prob=probability
                best_label=classval
            x_pred.append(best_label)
correct=0
for index,key in enumerate(x_test):
    if x_test[index][-1]==x_pred[index]:
        correct+=1
print("Accurary :",(correct/float(len(x_test)))*100)
