import pandas as pd
from collections import Counter
import math
tennis=pd.read_csv("third.csv")
print("\n GIVEN PLAY TENNIS DATA\n",tennis)
def entropy(alist):
    c=Counter(x for x in alist)
    instances=len(alist)
    prob=[x / instances for x in c.values()]
    return sum([-p * math.log(p,2) for p in prob])
def information(d,split,target):
    splitting=d.groupby(split)
    n=len(d.index)
    agent=splitting.agg({target :[entropy,lambda x:len(x)/n]})[target]
    agent.columns=['Entropy','Observation']
    newentropy=sum(agent['Entropy'] * agent['Observation'])
    oldentropy=entropy(d[target])
    return oldentropy - newentropy
def id3(sub,target,a):
    count=Counter(x for x in sub[target])
    if len(count)==1:
        return next(iter(count))
    else:
        gain=[information(sub,attr,target) for attr in a]
        print("Gain=",gain)
        maxi=gain.index(max(gain))
        best=a[maxi]
        print("BEST ATTRIBUTE",best)
        tree={best:{}}
        remain=[i for i in a if i!= best]
        for val,subset in sub.groupby(best):
            subtree=id3(subset,target,remain)
            tree[best][val]=subtree
        return tree
names=list(tennis.columns)
print("LIST OF ATTRIBUTES",names)
names.remove('playtennis')
print("predicting atrribute ",names)
tree=id3(tennis,'playtennis',names)
print("\n\n the resultant decision tree is \n")
print(tree)
