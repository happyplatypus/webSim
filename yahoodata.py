#!/usr/bin/python


import numpy as np
from pandas.io.data import DataReader


### define security universe
import pandas as pd



from functools import partial
from os.path import expanduser
home = expanduser("~")


sec=pd.read_csv(home+"/code/r_projects/investableUniverse/investableUniverse.csv")


sec1=sec[(sec.IPOyear<=2010)]

#print sec.IPOyear
sec1.shape

symbols_list=map(lambda x: x.upper(),sec.tic.tolist())
#symbols_list=map(lambda x: x.upper(),sec.tic.tolist())[1:50]


#print symbols_list

#for s in symbols_list:
#    print '"'+s+'"'

#print options.tic.tolist()


# In[ ]:

data=pd.read_pickle(home+'/code/python_projects/webSim/data.pickle')

#data.to_pickle('/home/puru/code/clojure_projects/clojure-download2/yahoodata.pickle')


# In[ ]:

#import requests
#r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
#r.status_code
from dateutil.parser import *


# In[ ]:


#symbols_list = list(set(options['tic'].tolist()))[0:10]
#symbols_list = list(set(options['tic'].tolist()))

#symbols_list = ['SPY','HSY']

if 0:
    d = []
    for ticker in symbols_list:
        print ticker
        tmp=DataReader(ticker, "yahoo", '2010-01-01')
    #tmp.columns=map(lambda x : ticker+"."+x,tmp.columns)
        tmp['tic']=ticker
        tmp['Date']=tmp.index
        d.append(tmp)
    data=pd.concat(tmp) 
    data.to_pickle(home+'/code/python_projects/webSim/data.pickle')

#data=d[0]
#data=data.join(d[1:],how='outer')






# In[ ]:

#data2=pd.merge(options,data,how='inner',on=['tic','Date'])

data.head()




#del data2['Date']

#print data2.tail(30)

#print len(list(set(data.tic)))
#print len(list(set(options.tic)))



# In[ ]:

#data.head()
#data.tail()

#"indicator 1.00 emaDiff Close 10 20"
#"indicator -1.00 emaDiff Volume 10 20"



def emaDiff(tic,series_,first,second):
    data1=data[data.tic==tic]
    #return pd.ewma(data1[series_],first)-pd.ewma(data1[series_],second)
    return data1[series_].ewm(first).mean()-data1[series_].ewm(second).mean()

def Diff(tic,series_,first,second):
    data1=data[data.tic==tic]
    #return pd.ewma(data1[series_],first)-pd.ewma(data1[series_],second)
    return data1[series_].shift(first)-data1[series_].shift(second)

def Ret(tic,series_,first,second):
    data1=data[data.tic==tic]
    #return pd.ewma(data1[series_],first)-pd.ewma(data1[series_],second)
    return np.log(data1[series_].shift(first))-np.log(data1[series_].shift(second))


#emaDiff("HSY",'Close' ,10, 20)
from sklearn.preprocessing import scale
#portfolio=symbols_list[0:500]
portfolio=symbols_list



print "calc weights"
#wts=map(partial(emaDiff,series_='Close' ,first=5, second=15),portfolio) ## un delayed
wts=map(partial(Diff,series_='Close' ,first=1, second=0),portfolio) ## un delayed


wts=pd.concat(wts,axis=1)
wts.columns=range(0,wts.shape[1])
tmp=wts.apply(np.absolute)
tmp=tmp.apply(np.sum,axis=1)
wts1=wts.div(tmp,axis=0)


#wts=map(partial(Diff,series_='Close' ,first=1, second=2),portfolio) ## un delayed
#wts=pd.concat(wts,axis=1)
#wts.columns=range(0,wts.shape[1])
#tmp=wts.apply(np.absolute)
#tmp=tmp.apply(np.sum,axis=1)
#wts=-wts.div(tmp,axis=0)


#wtsv=map(partial(emaDiff,series_='Volume' ,first=12, second=26),portfolio) ## un delayed
#wtsv=pd.concat(wtsv,axis=1)
#wtsv.columns=range(0,wtsv.shape[1])
#tmp=wtsv.apply(np.absolute)
#tmp=tmp.apply(np.sum,axis=1)
#wtsv=wtsv.div(tmp,axis=0)
#wtsv=wtsv*-1

#print wtsv.shape
print wts.shape
#wts=wts.add(wtsv)
#wts=wtsv



#print tmp
#print np.transpose(scale(np.transpose(wts.as_matrix()),with_std=False))
#wts_delayed=map(lambda x: x.shift(1),wts)



#map(lambda x,y:x*y,[1 ,2 ,3],[4, 5 ,6])
print "calc rets"
rets=map(partial(Ret,series_='Adj Close' ,first=0, second=1),portfolio)
rets=pd.concat(rets,axis=1)

final_returns=pd.DataFrame(rets.values*wts1.values,index=rets.index)
final_returns=final_returns.apply(np.nanmean,axis=1)
#rets
#final_returns=pd.concat(map(lambda x,y:x*y,wts,rets),axis=1).mean(axis=1)
#final_returns=rets.multiply(wts)
#wts
#final_returns.columns[0]='returns'
final_returns=final_returns[pd.notnull(final_returns)]
#final_returns['cumulative']=np.cumsum(final_returns)
#pd.DataFrame(final_returns)
final_returns2=pd.DataFrame()
final_returns2['cumulative']=np.cumsum(final_returns)

#print final_returns.tail(20)

#data.to_pickle('/home/puru/code/clojure_projects/clojure-download2/yahoodata.pickle')
#data.('/home/puru/code/clojure_projects/clojure-download2/yahoodata.pickle')

#final_returns2


# In[ ]:

wts.head()

PP=99
delay_=1
def calcReturn(ii):
	tmp=wts.ix[ii,:].copy()
	ul=np.nanpercentile(tmp,PP)
	ll=np.nanpercentile(tmp,100-PP)
	tmp[(tmp>ll) & (tmp<ul)]=0
	tmp1=tmp.apply(np.absolute)
	tmp1=tmp1.apply(np.sum,axis=0)
	tmp2=tmp.div(tmp1,axis=0)
	return np.sum(np.nan_to_num(tmp2.values*rets.ix[ii+delay_,:].values))



print "doing backtest"
N=wts.shape[0]
performance=map(calcReturn,range(N-1))
print np.cumsum(performance)

# In[ ]:

wts.head()


# In[ ]:



def logdiff(x,y):
    return np.log(x)-np.log(y)

def pdiff(x,y):
    return x/y-1




