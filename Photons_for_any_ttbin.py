
'''
Python for analysis of photon stream data from the Picoqunt GmbH Hydraharp and t3 data from Boris's labview program for the Swabian TimeTagger.

Adapted from photons.m V5.0 @ HENDRIK UTZAT, KATIE SHULENBERGER, BORIS SPOKOYNY, TIMOTHY SINCLAIR (10/29/2017)

Weiwei Sun, July, 2019


Update Sept. 2020, t2 data from swabian timetagger20 output files can be converted to .photons files and use functions in this class. (may need to rewrite main file though)


Update Feb. 2023, Colburn Cobb-Bruno, code adapted to use new file formats .ttbin provided by the lastest swabian timetagger 
'''



import numpy as np
import time as timing
import os, struct, scipy, warnings
from numba import jit
import matplotlib.pyplot as plt 
import TimeTagger
warnings.filterwarnings('ignore')
from array import array
import numba
from time import sleep

class Photons:
    
    
    def __init__(self, file_path, memory_limit = 1):
        
        # properties
        self.filereader = TimeTagger.FileReader(file_path)
        self.file_path = file_path
       # self.header_size = None # size of headers in bytes
        self.header = None # dictionary for header information
        self.memory_limit = memory_limit
        self.buffer_size = int(self.memory_limit * 1024 * 1024 / 8) # how many uint64 can be read at once
        self.cross_corr = None
        self.auto_corr = None     #  dictionary to store the correlations
        self.datatype = np.uint64 
        
       
        self.ch0 = None
        self.ch1 = None
        self.ch2 = None
        self.ch3 = None
        self.ch4= None
        
        # extract photon stream file info
        self.path_str = os.path.split(file_path)[0] #gets file path
        self.file_name = os.path.split(file_path)[1].split('.')[0] #gets name i.e. File is dot8.1.ttbin; returns dot8
        self.file_ext = os.path.split(file_path)[1].split('.')[1] # gets run i.e. File is dot8.1.ttbin; returns dot1
        self.g2 = None
        self.all_photon_data = None
        self.all_photon_data_no_OF = None
        self.intensity_counts = None
        self.sync_channel = None
        
        print('========================================')
        print('Photon class created')
        print('========================================')
        
       
   
    
        '''
        -----------------------------------------------------------------------------------
        This function puts all the data about channel, timestamp, and overflow information
        into an array to be written to binary 
        it also gets the header information containing the state of the .ttbin file at time of measurement. 
        you need to specify the measurementmode yourself
        -----------------------------------------------------------------------------------
        '''

        #n_events here is the size of the buffer to work with in order to speed up computational time
    
 
    def get_arival_data_and_header(self, measurement_mode, manual_resolution=0, n_events=1000000):
        start_time = timing.time()
        self.header = self.filereader.getConfiguration()
        self.header['MeasurementMode'] = measurement_mode
        inputs = self.header['inputs']
        
        if 'resolution rms' in self.header:
            resolution_rms = [d['resolution rms'] for d in inputs]
            self.header['Resolution'] = np.mean(resolution_rms)
        else:
            self.header['Resolution'] = manual_resolution
            
        filereader = TimeTagger.FileReader(self.file_path)
        
        # create empty arrays using np.empty()
        Complete_Channel_Array = np.empty(0, dtype=np.int64)
        Complete_Arrival_Time = np.empty(0, dtype=np.uint64)
        Complete_Overflow_Array = np.empty(0, dtype=np.uint8)
        Complete_Missed_Events = np.empty(0, dtype=np.uint32)
        data_list = [Complete_Channel_Array, Complete_Arrival_Time, Complete_Overflow_Array, Complete_Missed_Events]
    
        while filereader.hasData():
            data = filereader.getData(n_events=n_events)
            channel = data.getChannels()
            timestamps = data.getTimestamps()
            overflow_types = data.getEventTypes()
            missed_events = data.getMissedEvents()
            
            for i, arr in enumerate(data_list):
                data_list[i] = np.concatenate((arr, eval(f"{'channel' if i == 0 else 'timestamps' if i == 1 else 'overflow_types' if i == 2 else 'missed_events'}")))
        
        # concatenate arrays using np.concatenate()
        All_Photon_Data = np.concatenate([arr[np.newaxis, :] for arr in data_list], axis=0)
        
        # use np.transpose() instead of vstack()
        self.all_photon_data = All_Photon_Data
        self.all_photon_data_no_OF = self.all_photon_data[:2, :]
       
        del filereader
        
        end_time = timing.time()
        total_time = end_time - start_time
        print('Time elapsed for data header function is %4f s' % total_time)
        
    '''
    -----------------------------------------------------------------------------------
    this funciton writes all the data into an array with a single row
    for a file that you have specified to be t3 and using the channel the pulse enters,
    it will rewrite the file into an array that goes like [ch, t, tau, ch, t, tau...] t=absolute tau=time after pulse
    for t2 data it writes the data into an array that goes like [ch, t, ch, t...]
    if it is t2 data you can have the sync_channel anything
    ====================================================================================
    IMPORTANT:
        for t3 data I do not know what units self.header['SyncRate'] should be in as it 
        was given for all picoquant files but swabian has no such information tied to 
        their files. will be important to test if using t3 data
    ====================================================================================
    -----------------------------------------------------------------------------------
    '''
    def write_total_data_to_file(self, sync_channel = 1):
        time_start = timing.time()
        dir_path, file_name = os.path.split(self.file_path)
        new_file_name = os.path.splitext(file_name)[0] + ".photons"
        fout_file = os.path.join(dir_path, new_file_name)
        fout = open(fout_file, 'wb')
        
        relative_photon_data = np.array(self.all_photon_data_no_OF)  # if for some reason the array isnt in chronological order np.array([list(x) for x in zip(*sorted(zip(*self.all_photon_data_no_OF), key=lambda x: x[1]))])
        self.sync_channel = sync_channel
        
        if self.header['MeasurementMode'] == 2:
            flattened_array = np.zeros(self.all_photon_data_no_OF.shape[1]*2)
            for i in range(self.all_photon_data_no_OF.shape[1]):
                flattened_array[2*i] = self.all_photon_data_no_OF[0][i]
                flattened_array[2*i+1] = self.all_photon_data_no_OF[1][i]
          
              
            flattened_array.astype(self.datatype).tofile(fout)
         
            
         
        elif self.header['MeasurementMode'] == 3:
           
            
            mask = self.all_photon_data_no_OF[0,:] == sync_channel
            pulse_channels = self.all_photon_data_no_OF[:, mask]
            time_differences = np.diff(pulse_channels[1,:])
            self.header['SyncRate'] = 1/(np.mean(time_differences))
            
               
            mask = relative_photon_data[0,:] == sync_channel
            diffs = np.diff(np.where(np.append(False, mask)))[0]
            starts = np.where(mask)[0]
            
            for start, diff in zip(starts, diffs):
                relative_photon_data[1, start : start + diff] -= relative_photon_data[1, start]
         
            t3_type_array = np.delete(np.concatenate((self.all_photon_data_no_OF, relative_photon_data),axis = 0),2,0)
        #print (t3_type_array[:,:20])
        
            mask = t3_type_array[0,:] != sync_channel
            t3_type_array_no_pulse= t3_type_array[:,mask]
       # print(t3_type_array_no_pulse)
            flattened_array = np.zeros(t3_type_array_no_pulse.shape[1]*3)
        
            for i in range(t3_type_array_no_pulse.shape[1]):
               flattened_array[3*i] = t3_type_array_no_pulse[0][i]
               flattened_array[3*i+1] = t3_type_array_no_pulse[1][i]
               flattened_array[3*i+2] = t3_type_array_no_pulse[2][i]
        test = flattened_array
        flattened_array.astype(self.datatype).tofile(fout)
        
        fout.close()
        
       
        time_end = timing.time()
        total_time = time_end - time_start
        print('Time elapsed is %4f s' % total_time)


    '''
    ---------------------------------------------------
    this function below will create write all the photon timestamstamps into a binary file while making all the 
    channels =0
    use this as a check while also uncommenting all the relevant print statements in other functions
    
    photons1= Photons(file_name, 34)
    photons1.get_header_info(2)

    photons1.get_arival_data()
    a=photons1.all_photon_data_no_OF[1,-20:-1]
    for i in a.tolist():
        print("{:.10f}".format(i))
    print(len(photons1.all_photon_data_no_OF[0]))
    photons1.write_total_data_to_file()

    photons1.write_photons_to_one_channel('', '') is an example of what you would call
    ---------------------------------------------------
    '''

    def write_photons_to_one_channel(self, file_in):
    
        time_start = timing.time()
        counts = self.buffer_size * self.header['MeasurementMode']
        
        dir_path, file_name = os.path.split(self.file_path)
        in_file_name = os.path.splitext(file_name)[0] + ".photons"
        fin_file = os.path.join(dir_path, in_file_name)
        
        dir_path, file_name = os.path.split(self.file_path)
        out_file_name = os.path.splitext(file_name)[0] + '_ch0' + ".photons"
        fout_file = os.path.join(dir_path, out_file_name)
       
        fout = open(fout_file, 'wb')
        fin = open(fin_file, 'rb')
    
        while 1:
           batch = np.fromfile(fin, dtype=self.datatype, count = counts)
           batch[::self.header['MeasurementMode']] = 0 # set the channel number to be 0
           batch.tofile(fout)
    
           if len(batch) < counts:
               break
        
        
        ch0_arr = np.fromfile(fout_file, dtype=np.uint64 )
        self.ch0=ch0_arr
        
        fout.close()
        fin.close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('========================================')
        print('Photon records written to %s.photons' )
        print('Time elapsed is %4f s' % total_time)
        print('========================================')
        
        
       
        '''
        ---------------------------------------------------------------------------------
        This function writes the data in the .photons file to each respective channel
        photons_to_channel('','',# of channels(optional))
        for t3 mode, the the moment the laser sync channel is removed so it will write an empty file 
        for which ever channel the laser pulses were entering
        ---------------------------------------------------------------------------------
        '''

    def photons_to_channel(self, file_in, sub_dir, n_channel = 3):
    
        time_start = timing.time()
        counts = self.buffer_size * self.header['MeasurementMode']
        
        dir_path, file_name = os.path.split(self.file_path)
        in_file_name = os.path.splitext(file_name)[0] + ".photons"
        fin_file = os.path.join(dir_path, in_file_name)
  
        fin = open(fin_file, 'rb')
        
        dir_path, file_name = os.path.split(self.file_path)
        out_file_name = os.path.splitext(file_name)[0] + '_ch0' + '.photons'
        fout_file = [os.path.join(dir_path, out_file_name.replace('0', str(i+1))) for i in range(n_channel)]
        
        fout = [open(file, 'wb') for file in fout_file]
       
    
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            lbatch = len(batch)//self.header['MeasurementMode']
            batch.shape = lbatch, self.header['MeasurementMode']
            
            for i in range(1,n_channel+1):
                
                batch[batch[:, 0] == i].tofile(fout[i-1])
               
                
            if lbatch < self.buffer_size:
                break
       
        
        fin.close()
       
        for i in range(n_channel):
             
             fout[i].close() 
             
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)
        
        '''
       This function sorts photon data according to photon arrival time.
       For t2 data the time in ps is the absolute arrival time of the photons.
       For t3 data the time is relative to the sync pulse.
       A new .photons file is written containing the photons detected within tau_window: [lower_tau, upper_tau] in ps.
       =============================================================================
       IMPORTANT: 
           I have not really tried to fix this function but it doesnt seem 
       to work yet. I have gotten around the issue by using the np.sort function
       in any methods that require the photon stream to be in order which works well but it 
       may have limitations I havent found yet
       ==============================================================================
       '''
       
    def arrival_time_sorting(self, file_in, tau_window):
    
        time_start = timing.time()
        counts = self.buffer_size * self.header['MeasurementMode']
       
        dir_path, file_name = os.path.split(self.file_path)
        in_file_name = os.path.splitext(file_name)[0] + ".photons"
        fin_file = os.path.join(dir_path, in_file_name)
        
        dir_path, file_name = os.path.split(self.file_path)
        out_file_name = os.path.splitext(file_name)[0] + '_sorted' + ".photons"
        fout_file = os.path.join(dir_path, out_file_name)
       
        fin = open(fin_file, 'rb')
        fout = open(fout_file, 'wb')
      
    
    
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            lbatch = len(batch)//self.header['MeasurementMode']
            batch.shape = lbatch, self.header['MeasurementMode']
            ind_lower = batch[:, -1] > tau_window[0]
            ind_upper = batch[:, -1] <= tau_window[1]
            batch[ind_lower * ind_upper].tofile(fout)
    
            if lbatch < self.buffer_size:
                break
    
        fin.close()
        fout.close()
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)
    
    
    '''
    This function compiles and stores the intensity trace as a property of the photons class: self.intensity_counts.

    Two *args must be given, in the order of:
        file_in: filename of the .photons file without ending.
        bin_width: width of the bin for intensity compilation - ps for t2 data; number of pulses for t3 data.
    
    for plotting self.intensity_counts['trace'] is a matrix where each column is corresponds to each channel
    '''
           
    @jit
    def get_intensity_trace(self, file_in, bin_width):
    
        time_start = timing.time()
    
        dir_path, file_name = os.path.split(self.file_path)
        in_file_name = os.path.splitext(file_name)[0] + ".photons"
        fin_file = os.path.join(dir_path, in_file_name)
        fin = open(fin_file, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)
        fin.close()
    
        self.intensity_counts = {}
        length_photons = len(photons_records) // self.header['MeasurementMode']
        photons_records.shape = length_photons, self.header['MeasurementMode']
        n_bins = int(photons_records[-1,1] // bin_width)
        bins = np.arange(0.5, n_bins+1.5) * bin_width
        time_vec = np.arange(1, n_bins+1) * bin_width
        photon_trace = np.zeros((n_bins, 4)) # store the histogram
    
    
    
        for i in range(4):
            temp = photons_records[photons_records[:,0] == i,1]
            photon_trace[:, i] = np.histogram(temp, bins = bins)[0]
    
        self.intensity_counts['time'] = time_vec
        self.intensity_counts['bin_width'] = bin_width
        self.intensity_counts['trace'] = photon_trace
    
        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))
    
  
    '''
    This function histograms the lifetime of a .ht3 file with a given resolution.The histogram is stored as a property of the photons class: self.histo_lifetime.
    The given resolution should be a multiple of the original resolution used for the measurement. For instance, if the measurement resolution was 64 ps, then
    the resolution to form the histogram of the photon-arrival records could be 128, 256, 384, or 512 ps ...
    '''  
    
    
    def get_lifetime_histogram(self, file_in, resolution): 
    
        if self.header['MeasurementMode'] == 2:
            print('Only fot t3 data!')
            return False
        if resolution % int(self.header['Resolution']) != 0:
            print('The given resolution must be a multiple of the original resolution!\n')
            print('Check obj.header[\'Resolution\'].')
            return False
    
        time_start = timing.time()
        self.histo_lifetime = {}
    
        fin_file = self.path_str + file_in + '.photons'
        fin = open(fin_file, 'rb')
    
        # initializations
        rep_time = 1e12/self.header['SyncRate'] # in ps
        n_bins = int(rep_time//resolution)
        bins = np.arange(0.5, n_bins+1.5) * resolution
        print(bins)
        time = np.arange(1,n_bins+1) * resolution
        hist_counts = np.zeros(n_bins)
    
        counts = self.buffer_size * self.header['MeasurementMode']
        while 1:
            batch = np.fromfile(fin, dtype=self.datatype, count = counts)
            
            histo = np.histogram(batch[2::3], bins = bins)
            hist_counts +=  histo[0]
    
            if len(batch) < counts:
                break
        # This could be used to test whether we need batch operations
        # photons_records = np.fromfile(fin, dtype = self.datatype)
        # hist_counts = np.histogram(photons_records[2::3], bins = bins)[0]
    
        fin.close()
    
        #if self.header['Equip'] == 'TT':
            #time = rep_time - time
    
        self.histo_lifetime['Time'] = time
        self.histo_lifetime['Lifetime'] = hist_counts
        self.histo_lifetime['Resolution'] = resolution
    
        plt.semilogy(time/1000, hist_counts)
        plt.xlabel('Time [ns]')
        plt.ylabel('Counts')
        plt.title('Lifetime histogram with resolution ' + str(resolution) + ' ps')
        #plt.xlim(0,100)
        plt.show()
    
        time_end = timing.time()
        total_time = time_end - time_start
        print('Total time elapsed is %4f s' % total_time)
    



    '''
    ============================================================================================
    Photon correlation
    ============================================================================================
    '''

    '''
    Adapted from Boris Spokoyny's code.
    This function allows to correlate the photon-stream on a log timescale. The photon correlation is stored as a property of the photons class: self.cross_corr or self.auto_corr.

    file_in: file ID of the photons-file to correlate
    correlations: 'cross_corr' or 'auto_corr'.
    channels: Hydraharp channels to be correlated. e.g. [0,1] for cross-correlation of channels 0 and 1.
    time_bounds: upper and lower limit for the correlation. In ps for T2, in pulses for T3.
    lag_precision: Number of correlation points between time-spacings of log(2). Must be integers larger than 1.
    lag_offset: offset in ps or pulses between the channels.

    This algorithm computes the cross-correlation between ch0 and ch1 variables.
    For T2 data, ch0 and ch1 correspond to absolute photon arrival times.
    For T3 data, ch0 and ch1 should correspond to the photon arrival sync pulse number.
    start_time and stop_time for T2 data should be in time units of the photon arrival times, and for T3 data should be in units of sync pulses.

    The correlation lags are log(2) spaced with coarseness # of lags per cascade, i.e. if start_time = 1; stop_time = 50; coarseness = 4; the lag bins will be [1, 2, 3, 4;  6, 8, 10, 12;  16, 20, 24, 28;  36, 44 ]. If coarseness = 2, the lag bins become [1, 2;  4, 6;  10, 14;  22,30;  46].
    The algorithm works by taking each photon arrival time and counting the number of photons that are lag bins away. For example say our lag bin edges are [1, 2, 4, 6, 10, 14]
            Time Slot: 1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
            Photon     1   1   0   0   1   1   0   1   1   0   1   0   0   0   1
            1st Photon ^
            Lag bins   |   |       |       |               |               |
            # photons    1     0       2           2                1
            2nd Photon     ^
            Lag bins       |   |       |       |               |               |
            # photons        0     1       1           3               1
            3rd Photon                 ^
            Lag bins                   |   |       |       |               |               |
            # photons                    1     2       1           1               1

            etc..
    The cross-correlation is the sum of all the photons for each bin, i.e. for the three runs above we get [2, 3, 4, 6, 3].
  
    This function counts the number of photons in the photon stream bins according to a prescription from Ted Laurence: Fast flexible algirthm for calculating photon correlation, Optics Letters, 31, 829, 2006
    
    
    photon_corr give the G2 function which is the same as the normalized cross correlation
    '''
    
       
        
    @jit
    def photons_in_bins(self, ch0, ch1, lag_bin_edges, offset_lag):
        num_ch0 = len(ch0)
        num_ch1 = len(ch1)
        n_edges = len(lag_bin_edges)
        low_inds = np.zeros(n_edges, dtype = int)
        max_inds = np.zeros(n_edges, dtype = int)
        acf = np.zeros(n_edges-1)
    
        low_inds[0] = 1
        for phot_ind in range(num_ch0):
            bin_edges = ch0[phot_ind] + lag_bin_edges + offset_lag
    
            for k in range(n_edges-1):
                while low_inds[k] < num_ch1 and ch1[low_inds[k]] < bin_edges[k]:
                    low_inds[k] += 1
    
                while max_inds[k] < num_ch1 and ch1[max_inds[k]] <= bin_edges[k+1]:
                    max_inds[k] += 1
    
                low_inds[k+1] = max_inds[k]
                acf[k] += max_inds[k] - low_inds[k]
    
        return acf
    
    
    def photon_corr(self, file_in, correlations, channels, time_bounds, lag_precision, lag_offset = 0):
    
        time_start = timing.time()
    
        dir_path, file_name = os.path.split(self.file_path)
        in_file_name = os.path.splitext(file_name)[0] + ".photons"
        fin_file = os.path.join(dir_path, in_file_name) 
    
        fin = open(fin_file, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)

        length_photons = len(photons_records) // self.header['MeasurementMode']
        photons_records.shape = length_photons, self.header['MeasurementMode']
        fin.close()
        
        # split into channels
        ch0_u = photons_records[photons_records[:,0] == channels[0], 1] # ch0 syncs
        ch1_u = photons_records[photons_records[:,0] == channels[1], 1] # ch1 syncs
       
   
        
        ch0 = np.sort(ch0_u)
        ch1 = np.sort(ch1_u)
       
        
        start_time, stop_time = time_bounds
    
        '''create log 2 spaced lags'''
       
        cascade_end = int(np.log2(stop_time)) # cascades are collections of lags  with equal bin spacing 2^cascade
        nper_cascade =  lag_precision # number of equal
        a = np.array([2**i for i in range(1,cascade_end+1)]) #creates array of exponental spaced values
        b = np.ones(nper_cascade) #creates an array with the length lag_precision 
        division_factor = np.kron(a,b) 
        ''' division_factor creates an array that tells you how many bins you should have and at what spacing 
        this means that if you choose lag_precision of 3 then you would get [2,2,2,4,4,4,8,8,8...]
        '''
        lag_bin_edges = np.cumsum(division_factor/2)#creates an array as show above spaced like [1,2,3,5,7,9,13,17,21, 25...]
        lags = (lag_bin_edges[:-1] + lag_bin_edges[1:]) * 0.5# creates an array with the center values between each of the above points
    
        # find the bin region
        start_bin = np.argmin(np.abs(lag_bin_edges - start_time))
        stop_bin = np.argmin(np.abs(lag_bin_edges - stop_time))
        lag_bin_edges = lag_bin_edges[start_bin:stop_bin+1] # bins as shown above
        lags = lags[start_bin+1:stop_bin+1] # center of the bins as shown above
        division_factor = division_factor[start_bin+1:stop_bin+1] # normalization factor
    
    
        # counters etc for normalization
        ch0_min = np.inf
        ch1_min = np.inf # minimum time tag
        ch0_count = len(ch0)
        ch1_count = len(ch1) # photon numbers in each channel
        ch0_min = min(ch0_min, min(ch0))
        ch1_min = min(ch1_min, min(ch1))
    
        '''correlating '''
        tic = timing.time()
        print('Correlating data...\n')
    
        corr = self.photons_in_bins(ch0, ch1, lag_bin_edges, lag_offset)
    
        # normalization
        ch0_max = max(ch0)
        ch1_max = max(ch1)
        tag_range = max(ch1_max, ch0_max) - min(ch1_min, ch0_min) # range of tags in the entire dataset
    
        corr_div = corr/division_factor
        corr_norm = 2 * corr_div * tag_range**2 / (tag_range - lags)  / (ch0_count * ch1_count) # * ch0_max in boris' code. changed to tag_range
    
        print('Done\n')
        toc = timing.time()
        print('Time elapsed during correlating is %4f s' % (toc-tic))
    
        # store in property
        if self.header['MeasurementMode'] == 3:
            sync_period = 1e12/self.header['SyncRate']
            lags = lags * sync_period
    
        if 'cross' in correlations:
            self.cross_corr = {}
            self.cross_corr['lags'] = lags
            self.cross_corr['corr_norm'] = corr_norm
        elif 'auto' in correlations:
            self.auto_corr = {}
            self.auto_corr['lags'] = lags
            self.auto_corr['corr_norm'] = corr_norm
    
    
        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))

        '''
        This function calculates g2 for t3 data.
        file_in and channels are the same as photon_corr.
        time_range is the maximum time we're interested in, in ps.
        n_bins are the number of bins for correlation.
        '''

    def get_g2(self, file_in, channels, time_range, n_bins):
    
        if self.header['MeasurementMode'] == 2:
            print('Only for t3 data!')
            return False
    
        time_start = timing.time()
    
        fin_file = self.path_str +file_in + '.photons'
        fin = open(fin_file, 'rb')
        photons_records = np.fromfile(fin, dtype=self.datatype)
        length_photons = len(photons_records) // 3
        photons_records.shape = length_photons, 3
        fin.close()
       
        # split into channels
        pulse = 1e12 / self.header['SyncRate']
        ch0 = photons_records[photons_records[:,0] == channels[0], 2] + photons_records[photons_records[:,0] == channels[0], 1] * pulse# ch0 time
        
        ch1 = photons_records[photons_records[:,0] == channels[1], 2] + photons_records[photons_records[:,0] == channels[1], 1] * pulse# ch1 time
        
        
        
        # use linear spaced bins for g2 calculation
        bin_width = time_range // n_bins
        lag_bin_edges = np.arange(0, time_range + 2 * bin_width, bin_width)
        lags = np.hstack((-lag_bin_edges[-2::-1], lag_bin_edges[1:]))
        lag_bin_edges = lags- bin_width/2
    
    
        '''correlating '''
        tic = timing.time()
        print('Correlating data...\n')
    
        corr = self.photons_in_bins(ch0, ch1, lag_bin_edges, 0)
        #print(corr)
       
        # correct for offset
        n_ind = pulse // bin_width
        print(corr[n_bins:int(n_bins+1.5*n_ind)])
        ind_pulse_1 = np.argmax(corr[n_bins:int(n_bins+1.5*n_ind)]) # find the index of the first pulse
        offset = pulse - lags[ind_pulse_1+n_bins] # correct for offset
    
        print('Done\n')
        toc = timing.time()
        print('Time elapsed during correlating is %4f s' % (toc-tic))
        self.g2 = {}
        self.g2['lags'] = (lags[:-1] + offset) / 1e3 # in ns
        self.g2['g2'] = corr
    
        plt.plot(self.g2['lags'], self.g2['g2'], '-o', markersize = 1)
        plt.ylabel('Event counts')
        plt.xlabel('Pulse separation [ns]')
        
        plt.show()
    
        time_end = timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))



