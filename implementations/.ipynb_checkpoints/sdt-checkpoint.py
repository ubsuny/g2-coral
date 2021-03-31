# use a python program from Christoph Gohlke to read sdt files
# read a sdt file and get normalized g2 and time difference data as numpy arrays
# create a traing data set


from sdt_reader import sdtfile as sdt
import sys
import numpy as np

def read(x):
    """input the file name and output the raw g2 and time delay tau"""
    
    if x.endswith('.sdt'):
        file = sdt.SdtFile(x)
        tau = file.times[0]
        y = file.data[0]
    else:
        print('Please input a sdt file.')
        sys.exit()
    
    return tau, y


def normalize(t,y):
    """input raw data and output normalized data with zeroed time delay"""
    
    offset = t[np.argmin(y)]
    t -= offset     # set the dip to be a zero point for the time delay
    t *= 1e9       # convert time to ns
    
    ave = np.average(np.concatenate((y[:100],y[-100:])))
    y = y/ave 
    
    return(t,y)

def create_train(values, tstep):
    """given data and a time step, output training data set"""
    op1 = []
    #op2 = []
    for i in range(len(values) - tstep):
        op1.append(values[i : (i + tstep)])
        #op2.append(values[i + tstep])
    return np.stack(op1)
