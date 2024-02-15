# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:35:35 2023

@author: colbu
"""

import numpy as np
from matplotlib import pyplot as plt
import time as timing
import os
from Editied_Photons_for_any_TTBIN import Photons
import random
import math
from numba import jit

print('sim started')
#this function assigns photons to channels the arrival time array has been created 
@jit(nopython=True)
def process_photon_arr(photon_arr, p1, random_number, buffer_size):
    photon_number = photon_arr.shape[0]
    for buffer in range(1, int(photon_number / buffer_size) + 1):
        upper_index = buffer * buffer_size
        lower_index = (buffer - 1) * buffer_size

        temp_arr = p1[lower_index:upper_index].reshape(-1, 1)

        mask = random_number[lower_index:upper_index] > temp_arr

        temp_photon_arr = photon_arr[lower_index:upper_index]
        mask = np.squeeze(mask)

        temp_photon_arr[mask, 0] = 2
        photon_arr[lower_index:upper_index] = temp_photon_arr

    return photon_arr

#this function will assign frequencies to photons given two specific frequencies
@jit
def assign_freq_for_binaryfreq(photon_number,sequence_length, ensamble_size, freq_array,top_freq, bot_freq):
    for j in range(0,photon_number,int(sequence_length)):
        binomial_number = np.random.binomial( ensamble_size, .5, size=1)
        index0 = j
        index1 = int(j+sequence_length)
        
        prob = binomial_number/ ensamble_size
    
        for k in range(len(freq_array[index0:index1])):
            #change every other value to be 600 or 650c
            random_number = random.uniform(0,1)
         
            if random_number > prob:
                freq_array[index0+k] = bot_freq
            else:
                freq_array[index0+k] = top_freq
@jit
def assign_gaussian_frequency(photon_number, sequence_length, ensamble_size, freq_array, top_freq ,bot_freq, std_dev):
    temp = np.zeros((ensamble_size,int(photon_number/ensamble_size)))
    for emitter in range(ensamble_size):
        #np.random.randint(0, ensamble_size, ensamble_size) Is this the solution?
        for j in range(0,int(photon_number/ensamble_size),int(sequence_length/ensamble_size)):
            index0 = j
            index1 = int(j+sequence_length/ensamble_size)
            prob = np.random.binomial( 1, .5, size=1)
          
            random_number = random.uniform(0,1)
          
            if random_number > prob:
                temp[emitter,index0:index1] = np.random.normal(top_freq, std_dev, size=int(sequence_length/ensamble_size))
            else:
                temp[emitter,index0:index1] = np.random.normal(bot_freq, std_dev, size=int(sequence_length/ensamble_size))
    
    for photon in range(int(photon_number/ensamble_size)):
        rand_arr = np.random.randint(0, ensamble_size, ensamble_size)
        for i in range(ensamble_size):
            freq_array[photon*ensamble_size+i] = temp[rand_arr[i],photon]
'''
these two functions have different perfomances but essentially do the same thing. It is better to use the first with large photon numner usually
and better to use the second with small photon number and or small bin size. I would test them to see what works best for your combination of detector 
resolution and photon number. the second usually works better If you arent at or near the array memory limit for time_arr
'''
@jit(nopython=True)
def simulate_arrival_time(time_arr, PPS, total_time, detector_resolution, probability_per_bin):
    for i in range(1, PPS * total_time):
        passed_bins = 1
        while time_arr[i] == 0:
            random_number = random.random()

            if probability_per_bin > random_number:
                time_arr[i] = time_arr[i - 1] + passed_bins * detector_resolution
            else:
                passed_bins += 1
                
@jit(nopython=True)
def simulate_arrival_time_alternate(time_arr, PPS, total_time, detector_resolution, probability_per_bin, number_of_bins):
    for i in range(1, PPS * total_time):
        arrival_bin = random.randint(0, int(number_of_bins))
     
        time_arr[i] = arrival_bin*detector_resolution
    time_arr_sorted = np.sort(time_arr)
    return time_arr_sorted

#writes stage positions in an exponential manner
def exp_spacing_position(number_of_pos, final_pos, stage_position):
    
    x = final_pos**(1/number_of_pos)
    
    for i in range(0,number_of_pos+1):
        new_pos = x**i
        stage_position.append(new_pos)
#writes stage positions in a linear manner
def linear_spacing_position(number_of_pos, final_pos, stage_position):
    x = final_pos/number_of_pos
    for i in range(number_of_pos):
        a = i
        new_pos = x*a
        stage_position.append(new_pos)
        
correlation_arr = []
#what directory and filename to store the .photons file

directory = r"C:\Data and Code\Data\24_02_11_ensamble_PCFS_SIMS\Esize10"
#open the file for the position documentation
pos_file_name = directory + os.sep +'posFile.pos'
pcfs_file_name = directory + os.sep +'position.pcfslog'
pos_file = open(pos_file_name,'w')
pcfs_log_file = open(pcfs_file_name, 'w')
#choose center stage positions in nm as path length differences 
stage_position = []
number_of_pos = 100
final_pos = 5e6
linear_spacing_position(number_of_pos, final_pos,stage_position)


for i in stage_position:
      
    file_name = f'pos {int(i)}'
    
    ensamble_size = 10
    #photons per emitter per second
    photons_per_emitter = 1e7
    # Define time of simulation in s
    total_time =int(1)
    
    # Define number of photons per second
    PPS = int(photons_per_emitter) #int(ensamble_size*photons_per_emitter)
    photon_number =int(PPS*total_time)
    
    # Define detector resolution in ps
    detector_resolution = 1e2
    #the number of cycles you expect the frequency to go through in ps
    switching_time = 1e8
    expected_oscillations =  (total_time*1e12/switching_time)
   
    #create an array that switches between two values 
    sequence_length = photon_number/expected_oscillations  # Length of each sequence 
    
    print(f'Oscillation period is {sequence_length*(1/PPS)*(1e12)} ps')
    #freq_array =  [633 if i // sequence_length % 2 == 0 else 633.01  for i in range(photon_number)]
    freq_array = np.zeros(photon_number)
   
    top_freq = 709
    bot_freq = 709.4
    std_dev = .1
    start4 = timing.time()
    print('freq assignment started')
    assign_gaussian_frequency(photon_number, sequence_length, ensamble_size, freq_array, top_freq ,bot_freq, std_dev)
    end4 = timing.time()
    print(f'time to assign freqq {end4-start4}')
    #uncomment below for binary frequencies
    #assign_freq_for_binaryfreq(photon_number,sequence_length, ensamble_size, freq_array,top_freq, bot_freq)
   
    buffer_size = 500
  
    #choose a coherence length in nm
    #could be a variable function depending on environment
    coherence_length =1e10 #np.flip( np.linspace(1,50, photon_number))
    #center stage pos in nm as a path length difference
    center_stage_position = i
    #amplitude in nm /2 this is from center to peak 
    dither_amplitude = 1200
    #period in  picoseconds
    dither_time_period =int(1e12)
    time_between_stage_steps = 1e6
    #sync rate in Hz
    sync_rate = 100
    measurement_mode = 2
    
    
    start1 = timing.time()
    
    photon_arr = np.zeros((photon_number, 5))
    
    #set the 4th column to be photon color
    photon_arr[:,3]= freq_array
    #set the first column to be channel 1
    photon_arr[:,0] =1
    
    #set the 5th column to be photon coherence time
    photon_arr[:,4]=coherence_length
    
    #set the arrival time 

    # Calculate probability of finding a photon in a detector bin
    probability_per_bin = (PPS / 1e12) * detector_resolution
    number_of_bins = total_time*1e12/detector_resolution
    
    # Create arrival time array
    time_arr = np.zeros(PPS * total_time)
    '''
    should upgrade to do t3 data
    '''
    start1 = timing.time()
    #simulate_arrival_time(time_arr, PPS, total_time, detector_resolution, probability_per_bin)
    time_arr = simulate_arrival_time_alternate(time_arr, PPS, total_time, detector_resolution, probability_per_bin, number_of_bins)
    
    photon_arr[:,1] = time_arr
    end1 = timing.time()
    print(f'time to simulate arrival times {end1-start1}')
    #create dither waveform 
    #make sure the dither period is divisible by 4 to work with nice round numbers
    
    dither_waveform = np.zeros(int(dither_time_period/time_between_stage_steps))
   
    dither_waveform[:int(dither_time_period/(4*time_between_stage_steps))] = np.linspace(center_stage_position , dither_amplitude+center_stage_position,int(dither_time_period/(4*time_between_stage_steps)))
    dither_waveform[int(dither_time_period/(4*time_between_stage_steps)):int(3*dither_time_period/(4*time_between_stage_steps))] = np.linspace(dither_amplitude+center_stage_position , center_stage_position-dither_amplitude, int(dither_time_period/(2*time_between_stage_steps)))
    dither_waveform[int(3*dither_time_period/(4*time_between_stage_steps)):] = np.linspace(center_stage_position-dither_amplitude , center_stage_position,  int(dither_time_period/(4*time_between_stage_steps)))
    
    #Find where we are in the dither given absolute time
    relative_dither_time = photon_arr[:,1] % dither_time_period
    
    stage_pos = np.zeros(photon_number)
    
    #find the stage position given the photons time in relation to stage position
    for k in range(photon_number):
        index = int(relative_dither_time[k]/time_between_stage_steps)
     
        stage_pos[k] = dither_waveform[index]
        '''
        for a interferogram trace
        
        
        linear_motion = np.linspace(0,int(i),(int(1e6)))
        time_between_lin_steps = i*1e-8/total_time
        lin_index = (int(photon_arr[k,1])//time_between_lin_steps)
        stage_pos[k] = linear_motion[index]
        '''
    
    '''
    to construct the probalility of detector arrival
    assume stage position is at the peak of the interference
    '''
    
    p1 = .5 +.5*np.cos(4*np.pi*stage_pos[:]/photon_arr[:,3])#*np.exp(-np.abs(stage_pos[:])/photon_arr[:,4])
   
    random_number  = np.random.rand(photon_number, 1)

    start2 = timing.time()
    #assigns channels give probability, likely could be recast as a function above and forgo the buffering
    for buffer in range(1, int(photon_number/buffer_size)+1):
        upper_index = buffer*buffer_size
        lower_index = (buffer-1)*buffer_size
        
        temp_arr = p1[lower_index:upper_index].reshape(-1, 1)
        
        mask = random_number[lower_index:upper_index]>temp_arr
        
        temp_photon_arr = photon_arr[lower_index:upper_index]
        mask = np.squeeze(mask)
        
        temp_photon_arr[mask,0] = 2
        photon_arr[lower_index:upper_index] = temp_photon_arr
    
    end2 = timing.time()
    print(f'time to make photons array {end2-start2}')

    #plt.plot(p1)
    #plt.xlim(0,20000)
    #plt.show()
    
    #write positions to file as nm
    value = str(i*1e-6)
    
    pos_file.write(value + "\n")
    pcfs_log_file.write(value + "\n")
    
    
    
    #write data to .photons file
    if measurement_mode == 2:
        photons_mimic = photon_arr[:,:2]
        photons_mimic = photons_mimic.flatten()
      
        #write photons file in binary
        fout = open(directory +os.sep + file_name + '.photons', 'wb')
        
        photons_mimic.astype(np.uint64).tofile(fout)
        fout.close()
        
        '''
        all lines below are for data visualizaiton, comment out to run code faster
        '''
        photon1 = Photons(directory, 2, simulation = True, file_name ='pos 0', memory_limit = 1)
        
        photon1.get_arival_data_and_header(n_events = 1000000)
        
        photon1.write_total_data_to_file(sync_channel = 1)  
        
        #photon1.get_intensity_trace('24_02_11_ensamble_PCFS_SIMS' + os.sep + file_name, 1e10)
        
        #photon1.photon_corr('24_02_11_ensamble_PCFS_SIMS'  +os.sep + file_name, 'cross-correlation', [1,2], (1e1,1e15), 3)
        
      
        
        '''
        
        #graphing
        channel1 = [row[1] for row in photon1.intensity_counts['trace'] if row[1] != 0]
        time1 = np.linspace(0, len(channel1), len(channel1))
        
        channel2 = [row[2] for row in photon1.intensity_counts['trace'] if row[2] != 0]
        time2 = np.linspace(0, len(channel2), len(channel2))
        
        # Calculate min and max values for each channel
        min_values = [np.min(channel1), np.min(channel2)]
        max_values = [np.max(channel1), np.max(channel2)]
        
        # Calculate ratio of min:max for each channel
        ratios = [min_val / max_val for min_val, max_val in zip(min_values, max_values)]
        
        # Plot the channels
        plt.plot(time1, channel1, label='Channel 1')
        plt.plot(time2, channel2, label='Channel 2')
        
        # Add the ratio to the plot
        for i, ratio in enumerate(ratios):
            plt.text(0.95, 0.9 - i * 0.05, f'Extinciton ratio (Channel {i+1}): {ratio:.2f}',
                     transform=plt.gca().transAxes, ha='right')
        
        # Add plot titles and labels
        plt.title('Intensity Trace', fontsize=16)
        plt.xlabel('Time (*10ms)', fontsize=14)
        plt.ylabel('Photon Counts/10ms', fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=12)
        #legend = plt.legend(loc='upper left')
        #legend.get_texts()[0].set_text('Channel 1 (with LP filter)')
        #legend.get_texts()[1].set_text('Channel 2 (with extra mirror)')
        plt.ylim(min(min_values)-.5*min(min_values),max( max_values)+max( max_values)*.25)
        plt.xlim(0,50)
        #plt.setp(legend.get_texts(), fontsize=6)#change the legend size
        plt.show()
        
        
      
        plt.plot(photon1.cross_corr['lags'], photon1.cross_corr['corr_norm'], color='blue', linewidth=2)  # Adjust line color and thickness
        #plt.xlim(1e4, 1e14)
        plt.ylim(0, 2)
        plt.title(f"g^2(tau)  at pos {int(i)}nm and ensamble size {ensamble_size} " , fontsize=16)  # Increase font size of title
        plt.xlabel('Time (ps)', fontsize=14)  # Increase font size of x-axis label
        plt.ylabel('A.U.', fontsize=14)  # Increase font size of y-axis label
        plt.xscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add grid lines
        plt.tick_params(axis='both', which='major', labelsize=12)  # Increase font size of tick labels
        #plt.xlim(1e8,1e11)
        x_value = 1e6
        idx = np.argmin(np.abs(photon1.cross_corr['lags'] - x_value))
        x_value = photon1.cross_corr['lags'][idx]
        y_value = photon1.cross_corr['corr_norm'][idx]
        #plt.annotate(f'depth of g2: {y_value:.4f}', xy=(x_value, y_value), xytext=(x_value, y_value-.2))
                    
        
        plt.show()
        #correlation_arr.append( photon1.cross_corr['corr_norm'])
        '''
pos_file.close() 
pcfs_log_file.close()
#%%
#create colors for graph 
colors = [(.1,.6,.8),
        (0.2, 0.4, 0.8),  # Light blue
          (0.1, 0.2, 0.6),  # Medium blue
          (0.0, 0.1, 0.4),   #dark blue
          (0.9,0.2,0.2)] 

plt.plot(photon1.cross_corr['lags'],  correlation_arr[0], color=colors[3], linewidth=2,label = f'PLD : {int(stage_position[0]/1e3)}um')  # Adjust line color and thickness
plt.plot(photon1.cross_corr['lags'],  correlation_arr[1], color=colors[2], linewidth=2, label = f'PLD : {int(stage_position[1]/1e3)}um')
plt.plot(photon1.cross_corr['lags'],  correlation_arr[2], color=colors[1], linewidth=2, label = f'PLD : {int(stage_position[2]/1e3)}um')
plt.plot(photon1.cross_corr['lags'],  correlation_arr[3], color=colors[0], linewidth=2, label = f'PLD : {int(stage_position[3]/1e3)}um')
dash_pattern = [5,10]
plt.axvline(x = 1e8, color = colors[4], linestyle='--', dashes = dash_pattern, label='Switching Time')
plt.xlim(1e4, 1e12)
plt.ylim(0,2)
plt.title(f"g^2(tau) at ensamble size {ensamble_size} \n Integration time:{total_time}s \n 1e5 photons per second" , fontsize=16)  # Increase font size of title
plt.xlabel('Time (ps)', fontsize=14)  # Increase font size of x-axis label
plt.ylabel('A.U.', fontsize=14)  # Increase font size of y-axis label
plt.xscale('log')
#plt.grid(True, which='both', linestyle='--', alpha=0.5)  # Add grid lines
plt.tick_params(axis='both', which='major', labelsize=12)  # Increase font size of tick labels
#plt.xlim(1e8,1e11)
x_value = 1e6
idx = np.argmin(np.abs(photon1.cross_corr['lags'] - x_value))
plt.legend(loc = 'upper right')
x_value = photon1.cross_corr['lags'][idx]
y_value = photon1.cross_corr['corr_norm'][idx]
#plt.annotate(f'depth of g2: {y_value:.4f}', xy=(x_value, y_value), xytext=(x_value, y_value-.2))
            

plt.show()



#%%
'''
Check to see if the widths match the spectral convolution
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy
from scipy.stats import norm

# Define the parameters of the Lorentzian distribution
location1 = 0  # Location parameter (center)
scale1 = .1     # Scale parameter (full width at half maximum, FWHM)
location2 = .1 # Location parameter (center)
scale2 = .1
# Generate x-values
x = np.linspace(-10, 10, 2000)  # Adjust the range as needed




autocorrelation = []
for i in range(-1000,1000):
    pdf1a = norm.pdf(x, loc=location1, scale=scale1)
    pdf1b = norm.pdf(x, loc=location1-20/2000*i, scale=scale1)
    autocorrelation.append(sum(pdf1a*pdf1b))

plt.plot(x,autocorrelation/max(autocorrelation))
crosscorrelation = []
for i in range(-1000,1000):
    pdf1a = norm.pdf(x, loc=location2, scale=scale2)+norm.pdf(x, loc=location1, scale=scale1)
    pdf1b = norm.pdf(x, loc=location2-20/2000*i, scale=scale2)+norm.pdf(x, loc=location1-20/2000*i, scale=scale1)
    crosscorrelation.append(sum(pdf1a*pdf1b))

plt.plot(x,autocorrelation/max(autocorrelation),label = 'undiffused autocorr')
plt.plot(x,crosscorrelation/max(crosscorrelation),label  = 'diffused autocorr')
plt.xlim(-.7,.7)
plt.legend()
plt.show()
#%%
print((autocorrelation/max(autocorrelation))[983])
print((crosscorrelation/max(crosscorrelation))[981])

