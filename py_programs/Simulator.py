# This program generates mc simulated g2 correlation data,
# with customized 'N -> average detection events per bin'
# and g2(0) value based on the fraction of light sources: 
# The generated data will be saved as npy files.


# import libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import os
import sys
sys.path.append('./../data/')
import json

# create a simulator class
class simulator:
    
    Nbins = 0        # number of bins
    Ndetections = 0            # total detection events
    N = 0.           # average detection events per bin
    n_sets = 0         # number of data sets generated
    info = {}         # a dictionary containing simulation parameters
    # groundtruth = 0.   # g2(0) value, the threshold for classifiers
    # classify = False     # the binary result
    #bin_width = 1.    # bin width, in nanoseconds

    # fraction of light sources
    p_sps = 0.         
    p_laser = 0.      
    p_non = 0.       
    p_ther = 0.      
    
    
    # exp_rate = 1.       # exponential rate
    # n = 1           # number of identical single photon emitters
    # g20 = 1 - (1/n)  # correlation at zero time delay
    
    ''' 
    gt    ->  ground truth, g2(0)<gt will be considered as a single photon source
    width ->  bin width in nanoseconds
    '''

    
    def __init__(self,  Nbins,  Ndet, nset, sps, laser, non, ther):
        '''
        Nbins ->  number of bins
        Ndet  ->  total detection events
        nset   ->  number of data sets generated
        sps   ->  intensity fraction of a single photon source   
        laser ->  intensity fraction of coherent lasers      
        non   ->  fraction of no photo detected        
        ther  ->  intensity fraction of a thermal source 
        '''
        #self.groundtruth = gt
        self.Nbins = Nbins
        #self.bin_width = width
        self.Ndetections = Ndet
        self.N = Ndet / Nbins
        self.n_sets = nset 

        # normalizing source fraction
        self.p_sps = sps / (sps + non + ther + laser)
        self.p_laser = laser / (sps + non + ther + laser)
        self.p_non = non / (sps + non + ther + laser)
        self.p_ther = ther / (sps + non + ther + laser)
        
        self.info = {
            'single photon fraction' : self.p_sps,
            'laser fraction' : self.p_laser,
            'thermal source fraction': self.p_ther,
            'non-detected fraction': self.p_non,
            'number of bins': self.Nbins,
            'number of data sets': self.n_sets ,
            #'bin width (ns)': self.bin_width,
            'average detection events per bin': self.N,}

        Nbin = self.Nbins
        bin_array = tf.range(Nbin//2 * (-1) ,Nbin//2+1,delta=1,dtype=tf.float32)
        bin_n = tf.constant(np.linspace(min(bin_array),-1,Nbin//2),dtype=tf.float32)
        bin_p = tf.constant(np.linspace(1,max(bin_array),Nbin//2),dtype=tf.float32)
        
        # single photon source
        sps_dist = tfd.MixtureSameFamily(
            mixture_distribution = 
            tfd.Categorical(probs=[bin_n.shape[0]/(Nbin+0),0/Nbin, bin_p.shape[0]/(Nbin+0)]), # probability to get the second photon at a time slot
            components_distribution = 
            #tfd.Exponential(rate=exp_rate),
            tfd.Uniform(low=[min(bin_array),0.,1],high=[0,0.,max(bin_array)+1])
                       )
       
        # non-detected case
        non_dist = tfd.Uniform(low=10000.,high=10000.)
        
        # laser
        laser_dist = tfd.Uniform(low=min(bin_array),high=max(bin_array)+1)
        
        # thermal source
        ther_dist = tfd.MixtureSameFamily(
            mixture_distribution = 
            tfd.Categorical(probs=[bin_n.shape[0]/(Nbin+1),2/(Nbin+1), bin_p.shape[0]/(Nbin+1)]), # probability to get the second photon at a time slot
            components_distribution = 
            tfd.Uniform(low=[min(bin_array),0.,1],high=[0,1.,max(bin_array)+1])
                       )
        
        total_dist = tfd.Mixture(
            cat = tfd.Categorical(probs=[self.p_sps, self.p_laser, self.p_non, self.p_ther]),
            components = [sps_dist,laser_dist,non_dist,ther_dist] )
        
        self.total_dist = total_dist


        
        
    def piechart(self):
        '''
        output: a piechart of light sources
        '''
        fig,ax = plt.subplots(1,1)
        ax.pie([i for i in list(self.info.values())[:4]],
        labels=[i for i in list(self.info.keys())[:4]],autopct='%1.2f%%',startangle=90)
        ax.set_title('Probability of light sources')
        fig.show()
    

        
    def get_data(self, plot=False, save=False, name='data'):
        '''
        input: 
        plot -> boolean, it will make a histogram plot if true (default false)
        save -> boolean, will save data in a .txt file if true (default false)
        name -> filename, 'data' by default
        output:
        histogram plot, (optional)
        normalized g2 signal plot, (optional)
        bin number, normalized g2 signal, bin values, in ONE np.array, 
        data file, (optional)
        '''
        distribution = self.total_dist
        counts = self.Ndetections
        data = np.zeros((self.n_sets, 3, self.Nbins))  # output data shape, 3 here represents [binnumber, g2signal, binvalue] format

        for i in np.arange(self.n_sets):
            try:
                samples = distribution.sample(counts)
            except:
                raise Exception('Input distribution is incorrect, please check the light sources fraction.')
                break
        
            # adjust samples into numpy flattened arrays
            if type(samples)!=np.ndarray:
                samples = samples.numpy()
            samples = samples.flatten()

        
            # get histogram results
            Nbin = self.Nbins
            bin_array = tf.range(Nbin//2 * (-1) ,Nbin//2+1,delta=1,dtype=tf.float32)
            histogram = np.histogram(samples, bins=np.ndarray.tolist(bin_array.numpy()) )
            histvalue = histogram[0]
            binnumber = histogram[1][:-1]
            #binvalues = np.array
        
        
            # normalize the signal
            norm = (np.average(histvalue[:len(histvalue)//4]) + np.average(histvalue[len(histvalue)//4:]))/2
            #norm = np.average(histvalue[np.argsort(histvalue)[len(histvalue)//2:]]) 
            signal = histvalue/norm
            
            # assign results to data
            data[i] = np.array([binnumber, signal, histvalue])
            
        
        # make a histogram plot
        if plot:
            f1 = plt.figure(1)
            plt.hist(samples, bins=np.ndarray.tolist(bin_array.numpy()) )
            plt.title('detection histogram')
            plt.xlabel('delay time bin')
            
            f2 = plt.figure(2)
            plt.plot(binnumber,signal)
            plt.title('g2 signal')
            plt.xlabel('delay time bin')
            plt.ylabel('# events')
            #fig.tight_layout()
            print('Following plots are from the last set of data.')
        
        # save the data 
        if save: 
            filename = name + '.csv'
            if os.path.isfile(filename):
                raise Exception('File already exists, please use another name.')
            try: 
                file = open(filename,'w')  # 'a' for appending
            except:
                raise Exception('Please use valid name.')
                #file.write(json.dumps(self.info))
            
            # write in the simulation info 
            for key, value in self.info.items():    # iterate on info values
                file.write('%s:%10.4f\n' % (key, value))
                
            # write in data sets
            file.write('\n')
            file.write('data array format: [binnumber   normalized g2 signal   binvalues],  data ')
            np.savetxt(file, data[:,0,:], header='bin number array') 
            np.savetxt(file, data[:,1,:], header='normalized g2 signal') 
            np.savetxt(file, data[:,2,:], header='histogram values (int)') 
            file.close()
        
        return data
        
        
def load_data(filename):
    '''
    input: path + filename
    output: data sets, same format with generated data sets
    '''
    # I must say, this is a stupid way for parsing. I will switch to pandas once available
    parse = np.loadtxt(filename+'.csv', skiprows=9, max_rows=1000000)
    data = np.zeros((parse.shape[0]//3, 3, parse.shape[1]))
    data[:,0,:] = parse[:parse.shape[0]//3,:]
    data[:,1,:] = parse[parse.shape[0]//3:parse.shape[0]*2//3,:]
    data[:,2,:] = parse[parse.shape[0]*2//3:,:]
    
    return data
  
    
# get ground truth from data sets
def get_truth(data,thr):
    '''
    input: data array, threshhold for classifying sps or not sps
    output: a binary result array
    '''
    signal = data[:,1,:]
    binnumber = data[:,0,:]
    
    # create a 1d-array of g2(0) values
    g2zero = np.ndarray.flatten(np.array([signal[i][binnumber[i]==0] for i in range(signal.shape[0])]))
    binary = np.zeros(g2zero.shape)
    
    # if it's smaller than the threshold then make it to 1 (it is a sps)
    binary[[g2zero[i]<thr for i in range(len(g2zero))]] = 1
    
    return binary
        
        
     