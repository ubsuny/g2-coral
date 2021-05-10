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
sys.path.append('./data/')
import json


# create a simulator class

class simulator:
    Nbins = 0        # number of bins
    bin_width = 1.    # bin width, in nanoseconds
    Ndetections = 0            # total detection events
    N = 0.           # average detection events per bin
    ndata = 0         # number of data files generated
    info = {}         # a dictionary containing simulation parameters
    groundtruth = 0.   # g2(0) value, the threshold for classifiers
    classify = False     # the binary result
    
    # fraction of light sources
    p_sps = 0.         
    p_laser = 0.      
    p_non = 0.        
    p_ther = 0.      
    
    
    
   # exp_rate = 1.       # exponential rate
   # n = 1           # number of identical single photon emitters
    # g20 = 1 - (1/n)  # correlation at zero time delay
    

    
    def __init__(self, gt, Nbins, width, Ndet, sps, laser, non, ther):
        '''
        gt    ->  ground truth, g2(0)<gt will be considered as a single photon source
        Nbins ->  number of bins
        width ->  bin width in nanoseconds
        Ndet  ->  total detection events
        sps   ->  intensity fraction of a single photon source   
        laser ->  intensity fraction of coherent lasers      
        non   ->  fraction of no photo detected        
        ther  ->  intensity fraction of a thermal source 
        '''
        self.groundtruth = gt
        self.Nbins = Nbins
        self.bin_width = width
        self.Ndetections = Ndet
        self.N = Ndet / Nbins
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
            'bin width (ns)': self.bin_width,
            'average detection events per bin': self.N,}
        
        
    def piechart(self):
        '''
        output a piechart of light sources
        '''
        fig,ax = plt.subplots(1,1)
        ax.pie([i for i in list(self.info.values())[:4]],
        labels=[i for i in list(self.info.keys())[:4]],autopct='%1.2f%%',startangle=90)
        ax.set_title('Probability of light sources')
        fig.show()
        
    #total_dist = 0.
    def distribution(self):
        '''
        output a tf distribution about the probability distribution of co-detection events
        '''
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
        
        #self.total_dist = total_dist
        return total_dist

        
    def get_data(self,dist, plot=False, save=False, name='data'):
        '''
        input: 
        distribution -> a tf distribution,  
        plot -> boolean, it will make a histogram plot if true (default false)
        save -> boolean, will save data in a .txt file if true (default false)
        name -> filename, 'data' by default
        output:
        histogram plot,
        normalized g2 signal plot,
        bin number and signal in np.array,
        result in binary: 1 -> it's single photon source; 0 -> it's not
        '''
        distribution = dist
        counts = self.Ndetections
        
        try:
            samples = distribution.sample(counts)
        except:
            return 'please check your input distribution'
        
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
        norm = np.average(histvalue[np.argsort(histvalue)[len(histvalue)//2:]]) 
        signal = histvalue/norm
        
        
        # make a histogram plot
        if plot:
            f1 = plt.figure(1)
            plt.hist(samples, bins=np.ndarray.tolist(bin_array.numpy()) )
            plt.title('mixed sources detection histogram')
            plt.xlabel('delay time bin')
            
            f2 = plt.figure(2)
            plt.plot(binnumber,signal)
            plt.title('mixed sources detection plot')
            plt.xlabel('delay time bin')
            plt.ylabel('# events')

            #fig.tight_layout()
        
        
        # check it is a single photon source or not
        g2zero = signal[binnumber==0.]
        if g2zero < self.groundtruth:
            self.classify = True
        else:
            self.classify = False
        
        binary = 1 if self.classify else 0
        
        
        # return values
        data = np.array([signal, binnumber])
        
        
        # save the data 
        if save: 
            filename = './data/' + name + '.txt'
            
            if os.path.isfile(filename):
                raise Exception('file already exists, please use another name')
            try: 
                file = open(filename,'w')  # 'a' for appending
            except:
                raise Exception('please use valid name')
                #file.write(json.dumps(self.info))
            
            for key, value in self.info.items():    # iterate on info values
                file.write('%s:%10.4f\n' % (key, value))
            file.write('\n')
            np.savetxt(file, np.array([binary]), header='binary result', delimiter=",")     # add bin numbers
            file.write('\n')
            np.savetxt(file, signal, header='g2 signal', delimiter=',')   # append g2 values
            file.write('\n')
            np.savetxt(file, binnumber, header='time bin value', delimiter=",")     # add bin numbers
            file.close()
        
        return data, binary
        
        
        
        
    #def histogram(self,samples):
        '''
        input:
        samples -> numpy or tensorflow array
        output:
        histogram values, bin values
        histogram values, bin values
        '''
        
        '''if type(samples)!=np.ndarray:
            samples = samples.numpy()
        samples = samples.flatten()
        
        Nbin = self.Nbins
        bin_array = tf.range(Nbin//2 * (-1) ,Nbin//2+1,delta=1,dtype=tf.float32)
        histogram = np.histogram(samples, bins=np.ndarray.tolist(bin_array.numpy()) )
        histvalue = histogram[0]
        binnumber = histogram[1][1:]
         
        if plot:
            plt.hist(samples, bins=np.ndarray.tolist(bin_array.numpy()) )
        
        return histvalue, binnumber'''
                
    # write data into a .txt file
    def savedata(self, histvalue, binnumber, name):
        '''
        input: histogram values, bin number, file name in a string
        '''
        filename = './data/' + name + '.txt'
        try: 
            file = open(filename,'a')
        except:
            raise Exception('file already exists, please use another name')
        #file.write(json.dumps(self.info))
        
        for key, value in self.info.items():    # iterate on info values
            file.write('%s:%10.4f\n' % (key, value))
        file.write('\n')
        np.savetxt(file, np.array([binary]), header='binary result', delimiter=",")     # add bin numbers
        file.write('\n')
        np.savetxt(file, signal, header='g2 signal', delimiter=',')   # append g2 values
        file.write('\n')
        np.savetxt(file, binnumber, header='time bin value', delimiter=",")     # add bin numbers
        file.close()
        