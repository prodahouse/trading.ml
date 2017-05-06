# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 03:15:00 2016

@author: Owen
"""

import math as m

# calculate the population standard deviation
def stdev_p(data):
    datamean=m.fsum(data)/len(data)    
    result = 0
    for i in test:
        result+=m.pow(i-datamean, 2)
    result = m.sqrt(result/len(data))
    return result

# calculate the sample standard deviation
def stdev_s(data):
    datamean=m.fsum(data)/len(data)    
    result = 0
    for i in test:
        result+=m.pow(i-datamean, 2)
    result = m.sqrt(result/(len(data)-1))
    return result

if __name__ == "__main__":
    test = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    #len(test)
    print "the population stdev is", stdev_p(test)
    print "the sample stdev is", stdev_s(test)