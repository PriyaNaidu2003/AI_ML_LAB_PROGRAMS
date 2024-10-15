import csv
cf=open('data2.csv','r')
r=csv.reader(cf)
a=[]
for rr in r:
    a.append(rr)
    print(rr)
n_att=len(a[0])-1
print("initail hypo is")
s=['0']*n_att
g=['?']*n_att
print("the most  specific:",s)
print("the most  general:",g)
for j in range(0,n_att):
    s[j]=a[0][j]
print("The candidate algo\n")
t=[]
for i in range(0,len(a)):
    if(a[i][n_att]=='yes'):
        for j in range(0,n_att):
            if(a[i][j]!=s[j]):
                s[j]='?'
        for j in range(0,n_att):
            for k in range(1,len(t)):
                if(t[k][j]!='?' and t[k][j]!=s[j]):
                    del t[k]
        print("for instance{0} the hypothesis is s{0}".format(i+1),s)
        if(len(t)==0):
            print("for instance {0} the hypothesis is g{0}".format(i+1),g)
        else:
            print("for instance {0} the hypothesis is g{0}".format(i+1),t)
    if(a[i][n_att]=='no'):
        for j in range(0,n_att):
            if(s[j]!=a[i][j] and s[j]!='?'):
                g[j]=s[j]
                t.append(g)
                g=["?"]*n_att
    print("for instance {0} the hypothesis is s{0}".format(i+1),s)
    print("for instance {0} the hypothesis is g{0}".format(i+1),t)
        
