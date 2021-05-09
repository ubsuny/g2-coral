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

# create a simulator class

class simulator:
    Nbins = 0        # number of bins
    bin_width = 1.    # bin width, in nanoseconds
    Ndetections = 0            # total detection events
    N = 0.           # average detection events per bin
    ndata = 0         # number of data files generated
    
    # fraction of light sources
    p_sps = 0.         
    p_laser = 0.      
    p_non = 0.        
    p_ther = 0.      

    
   # exp_rate = 1.       # exponential rate
   # n = 1           # number of identical single photon emitters
    # g20 = 1 - (1/n)  # correlation at zero time delay
    

    
    def __init__(self, Nbins, width, Ndet, sps, laser, non, ther):
        '''
        Nbins ->  number of bins
        width ->  bin width in nanoseconds
        Ndet  ->  total detection events
        sps   ->  intensity fraction of a single photon source   
        laser ->  intensity fraction of coherent lasers      
        non   ->  fraction of no photo detected        
        ther  ->  intensity fraction of a thermal source 
        '''
        self.Nbins = Nbins
        self.bin_width = width
        self.Ndetections = Ndet
        self.N = Ndet / Nbins
        self.p_sps = sps / (sps + non + ther + laser)
        self.p_laser = laser / (sps + non + ther + laser)
        self.p_non = non / (sps + non + ther + laser)
        self.p_ther = ther / (sps + non + ther + laser)
        
    def info(self):
        info = {
            'single photon fraction' : self.p_sps,
            'laser fraction' : self.p_laser,
            'thermal source fraction': self.p_ther,
            'non-detected fraction': self.p_non,
            'number of bins': self.Nbins,
            'bin width (ns)': self.bin_width,
            'average detection events per bin': self.N,}
        return info
        
        
    def piechart(self):
        fig,ax = plt.subplots(1,1)
        ax.pie([i for i in list(info.values())[:4]],
        labels=[i for i in list(info.keys())[:4]],autopct='%1.2f%%',startangle=90)
        ax.set_title('Probability of light sources')
        fig.show()
        

    def distribution(self):
        '''
        generate probability distribution of co-detection events
        '''
        Nbin = self.Nbins
        bin_array = tf.range(Nbin//2 * (-1) ,Nbin//2+1,delta=1,dtype=float32)
        bin_n = tf.constant(np.linspace(min(bin_array),-1,Nbin//2),dtype=float32)
        bin_p = tf.constant(np.linspace(1,max(bin_array),Nbin//2),dtype=float32)
        
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
        
        
        
        
        
        
        
        
        
        
        
   