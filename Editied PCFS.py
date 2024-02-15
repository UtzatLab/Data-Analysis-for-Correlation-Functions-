
import numpy as np
import time as timing
import os, struct, scipy, re, glob
from Editied_Photons_for_any_TTBIN import Photons 
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate


def extract_number(s):
    return int(s.split()[-1])


class PCFS:

    '''
    Called when creating a PCFS object. This function saves the fundamental arguments as object properties.
    folder_path is the full path to the folder containin the photon stream data and meta data of the PCFS run. NOTE: '\' needs to be replaced to '\\' or '/' and the not ending with '/', i.e., 'D/Downloads/PCFS/DotA'.
    memory_limit is the maximum memory to read metadata at once, set to default 1 MB.
    '''
    def __init__(self, folder_path ,measurement_mode, simulation = False, memory_limit = 1, header_lines_pcfslog = 5):

        tic = timing.time()
        # property
        self.simulation = simulation
        self.cross_corr_interferogram = None
        self.auto_corr_sum_interferogram = None
        self.tau = None
        self.PCFS_interferogram = None
        self.blinking_corrected_PCFS_interferogram = None
        self.spectral_correlation = {}
        self.Fourier = {}
        self.memory_limit = memory_limit
        self.measurement_mode = measurement_mode
        # extract files information
        self.path_str = folder_path
        self.PCFS_ID = os.path.split(folder_path)[1] # extract the folder name, i.e., 'DotA'
        os.chdir(folder_path)
        file_pos = glob.glob('*.pos')
        if len(file_pos) == 0:
            print('.pos file not found!')
      
        self.file_pos = file_pos[0] # with extension
        self.file_number = len(file_pos)
        file_pcfslog = glob.glob('*.pcfslog')
        if len(file_pcfslog) == 0:
            print('.pcfslog file not found!')
        self.file_pcfslog = file_pcfslog[0] # with extension
        if simulation:
            self.file_stream = [f.replace('.photons', '') for f in glob.glob('*.photons')]
            
        else:
            self.file_stream = [f.replace('.1.ttbin', '') for f in glob.glob('*.1.ttbin')] # without extension
   


        # read in the position file as array
        self.get_file_photons() # get all the photons files in the current directory
        self.stage_positions = np.loadtxt(self.file_pos)
        
        # read in the metadata of the .pcfslog file and store it as property
        with open(self.file_pcfslog) as f:
            lines_skip_header = f.readlines()[header_lines_pcfslog:]
        self.pcfslog = {}
        for lines in lines_skip_header:
            lines_split = lines.split('=')
            if len(lines_split) == 2:
                self.pcfslog[lines_split[0]] = float(lines_split[1])

        #create photons object for the photon stream at each interferometer path length difference.
        #self.file_stream is 
        self.photons = {}
       
        for f in self.file_stream:
        
            if simulation:
                self.photons[f] = Photons(folder_path+ os.sep+ f+'.photons',self.measurement_mode, self.simulation, self.memory_limit)
            else:
                self.photons[f] = Photons(folder_path+ os.sep+ f+'.1.ttbin',self.measurement_mode, self.simulation, self.memory_limit)
       
        toc= timing.time()
        print('Total time elapsed to create PCFS class is %4f s' % (toc - tic))


    '''
    ============================================================================================
    Get and parse photon stream / get correlation functions
    '''


    '''
    This function gets all the photon files in the current directory.
    self.file_photons is a list of the filenames without .photon at the end 
    '''
    def get_file_photons(self):
     
        temp = [f.replace('.photons','') for f in glob.glob('*.photons')] # without extension
        self.file_photons = sorted(temp, key=extract_number)
        print(self.file_photons)
        
        


    '''
    This function gets all photon stream data.
    '''
    def get_photons_all(self):

        time_start = timing.time()
        self.get_file_photons()
       
        for f in self.file_stream:
            if f not in self.file_photons:
                self.photons[f].get_arival_data_and_header()
                self.photons[f].write_total_data_to_file()
        time_end= timing.time()
        print('Total time elapsed to get all photons is %4f s' % (time_end - time_start))



    '''
    This function gets the sum signal of the two detectors for all photon arrival files.
    '''
    def get_sum_signal_all(self):
        time_start = timing.time()
        self.get_file_photons()
        
      
        for f in self.file_photons:
           
            if 'sum' not in f and ('sum_signal_' + f) not in self.file_photons :
              
                self.photons[f].write_photons_to_one_channel(f, 'sum_signal_'+f)
               
        time_end= timing.time()
        print('Total time elapsed to get sum signal of photons is %4f s' % (time_end - time_start))



    '''
    This function obtains the cross-correlations and the auto-correlation of the sum signal at each stage position and returns them in an array self.cross_corr_interferogram and the auto-correlation function of the sum signal to an array of similar structure.
    '''
    def get_intensity_correlations(self, time_bounds, lag_precision):
        time_start= timing.time()
        self.get_file_photons()
        self.time_bounds = time_bounds
        self.lag_precision = lag_precision
        cross_corr_interferogram = None
        auto_corr_sum_interferogram = None

        if len(self.file_photons) == len(self.file_stream):
            self.get_sum_signal_all()
        self.get_file_photons()
        
        i = -1
        j= -1
        for f in self.file_photons:
            # looking to get cross correlation
            
            if 'sum' not in f:
                i+=1
                if self.photons[f].cross_corr is None:
                    self.photons[f].photon_corr(f, 'cross', [1,2], time_bounds, lag_precision, 0)
                if self.tau is None:
                    self.tau = self.photons[f].cross_corr['lags']
                    self.length_tau = len(self.tau)

                # create an array containing to be filled up with the PCFS interferogram.
                if cross_corr_interferogram is None:
                    cross_corr_interferogram = np.zeros((self.length_tau, len(self.stage_positions)))
                if self.simulation:
                    correlation_number = self.file_number
                else:    
                    correlation_number = int(len(glob.glob('*.1.ttbin'))) # extract the number of correlation measurements from the file names
             
                ind =np.abs((correlation_number-1)-i)
            
                cross_corr_interferogram[:, ind] = self.photons[f].cross_corr['corr_norm']
               
            # looking to get auto-correlation for sum signals
            else:
                j +=1
                if self.photons[f[11:]].auto_corr is None:
                    self.photons[f[11:]].photon_corr(f, 'auto', [0,0], time_bounds, lag_precision, 0)
                if self.tau is None:
                    self.tau = self.photons[f[11:]].auto_corr['lags']
                    self.length_tau = len(self.tau)
                # create an array containing to be filled up with the PCFS interferogram.
                if auto_corr_sum_interferogram is None:
                    auto_corr_sum_interferogram = np.zeros((self.length_tau, len(self.stage_positions)))
                if self.simulation:
                    correlation_number = self.file_number
                else:    
                    correlation_number = int(len(glob.glob('*.1.ttbin')))
                
                ind =np.abs((correlation_number-1)-j)# extract the number of correlation measurements from the file names
                auto_corr_sum_interferogram[:, ind] = self.photons[f[11:]].auto_corr['corr_norm']
                
            print('==============================')

        self.cross_corr_interferogram = cross_corr_interferogram.copy()
        self.auto_corr_sum_interferogram = auto_corr_sum_interferogram.copy()
        
        # substract auto-correlation of sum signal from the cross correlation.
        PCFS_interferogram = cross_corr_interferogram - auto_corr_sum_interferogram
        self.PCFS_interferogram = PCFS_interferogram.copy()


        time_end= timing.time()
        print('Total time elapsed is %4f s' % (time_end - time_start))



    '''
    ============================================================================================
    Analysis of data
    '''


    '''
    This function gets blinking corrected PCFS interferogram.
    '''
    def get_blinking_corrected_PCFS(self):
        
        if self.simulation:
            self.blinking_corrected_PCFS_interferogram = 1 - self.cross_corr_interferogram
        else:
            self.blinking_corrected_PCFS_interferogram = 1 - self.cross_corr_interferogram / self.auto_corr_sum_interferogram
     

    '''
    This function gets and plots spectral diffusion tau_select units should be in ps
    '''
    def plot_spectral_diffusion(self, tau_select, white_fringe):
        if self.get_blinking_corrected_PCFS is None:
            self.get_blinking_corrected_PCFS()

        # plot PCFS interferogram at different tau
        x = 2 * (self.stage_positions - white_fringe) # in mm
  
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e6 for tau in tau_select] # in us
        y = self.blinking_corrected_PCFS_interferogram[ind, :]
     
        x = x/(3e8)/1000*1e12 # convert to ps
     
        plt.figure()
        plt.subplot(3,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$g^{(2)}_{cross} - g^{(2)}_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' PCFS Interferogram at $\tau$ [us]')

        plt.subplot(3,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'$g^{(2)}_{cross} - g^{(2)}_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)
        plt.title('Normalized ' + self.PCFS_ID + r' PCFS Interferogram at $\tau$ [us]')

        plt.subplot(3,1,3)
        for i in range(len(tau_select)):
            plt.plot(x, np.sqrt(y[i,:]/max(y[i,:])))

        plt.ylabel(r'$g^{(2)}_{cross} - g^{(2)}_{auto}$')
        # plt.xlabel('Optical Path Length Difference [mm]')
        plt.xlabel('Optical Path Length Difference [ps]')
        plt.legend(legends)

        plt.title('Squared root of Normalized ' + self.PCFS_ID + r' PCFS Interferogram at $\tau$ [us]')
        plt.tight_layout()
        plt.show()


    '''
    This function fits the autocorrelation of the sum of the interferogram and fits to the FCS traces, and creates an array with the fitted curves, parameters, and residuals.
    '''
    def fit_FCS_traces(self, interferogram, tau):
        f = lambda x, *p: p[0] + p[3] /(1 + x / p[1]) / np.sqrt(1 + x / p[1] / p[2] / p[2])
        tau_end = np.where(interferogram[:, i] == 0)[0][0]
        tau_select = tau[:tau_end]
        auto_corr = interferogram[:tau_end, i]
        p0 = [min(auto_corr), 3e9, 1, max(auto_corr)]
        p = curve_fit(f, tau_select, auto_corr, p0)
        p_fit = p[0]
        fit_curve = f(tau_select, p_fit)
        residuals = np.sum((auto_corr - fit_curve)**2)
        return p_fit, fit_curve, residuals




    '''
    ==========================================================================================
    Others
    '''


    '''
    This function gets spectral correlation data.
    '''
    def plot_mirror_spectral_corr(self, tau_select, xlim):
        x = self.mirror_spectral_correlation['zeta']
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e9 for tau in tau_select]
        y = self.mirror_spectral_correlation['spectral_corr'][ind,:]
      
        plt.figure()
        plt.subplot(2,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')

        plt.subplot(2,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)

        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')
        plt.tight_layout()
        plt.show()



    '''
    This function gets mirrored spectral correlation by interpolation.
    '''
    def get_mirror_spectral_corr(self, white_fringe_pos, white_fringe_ind):
        end = -1
        # construct mirrored data
        
        interferogram = self.blinking_corrected_PCFS_interferogram[:,:end]
     
        mirror_intf = np.hstack((np.fliplr(interferogram[:, white_fringe_ind:]), interferogram[:, white_fringe_ind+1:]))
     
        temp = white_fringe_pos - self.stage_positions[white_fringe_ind:end]
     
        temp = temp[::-1]
        mirror_stage_pos = np.hstack((temp, self.stage_positions[white_fringe_ind+1:end] - white_fringe_pos))
     
        interp_stage_pos = np.arange(min(mirror_stage_pos), max(mirror_stage_pos)+0.1, 0.1 )
        
        # row-wise interpolation
        a,b = mirror_intf.shape
        interp_mirror = np.zeros((a,len(interp_stage_pos)))
       
        for i in range(a):
            interp_mirror[i,:] = np.interp(interp_stage_pos, mirror_stage_pos, mirror_intf[i,:])
        
        self.mirror_stage_positions = mirror_stage_pos
        self.mirror_PCFS_interferogram = interp_mirror # not including the first line of position
 
        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(interp_stage_pos)
        path_length_difference = 0.2 * (interp_stage_pos) # NOTE: This is where we convert to path length difference space in cm.
    
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # zeta_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = self.mirror_PCFS_interferogram.copy()
        for i in range(a):
            spectral_correlation[i,:] = np.abs(np.fft.fftshift(np.fft.fft(self.mirror_PCFS_interferogram[i,:])))
            
        self.mirror_spectral_correlation = {}
        self.mirror_spectral_correlation['spectral_corr'] = spectral_correlation
        self.mirror_spectral_correlation['zeta'] = zeta_eV
        
    '''
    Using spline interpolation
    '''
    def get_splev_mirror_spec_corr(self, white_fringe_pos, white_fringe_ind,stage_increment=0.005):

        # construct mirrored data
        interferogram = self.blinking_corrected_PCFS_interferogram[:,:]
        mirror_intf = np.hstack((np.fliplr(interferogram[:, white_fringe_ind:]), interferogram[:, white_fringe_ind+1:]))
        temp = white_fringe_pos - self.stage_positions[white_fringe_ind:]
        temp = temp[::-1]
        mirror_stage_pos = np.hstack((temp, self.stage_positions[white_fringe_ind+1:] - white_fringe_pos))
        interp_stage_pos = np.arange(min(mirror_stage_pos), max(mirror_stage_pos)+stage_increment, stage_increment )

        # row-wise interpolation
        a,b = mirror_intf.shape
        interp_mirror = np.zeros((a,len(interp_stage_pos)))
        for i in range(a):
            x = mirror_stage_pos
            y = mirror_intf[i,:]
            tck = interpolate.splrep(x, y, s=0)
            xnew = interp_stage_pos
            ynew = interpolate.splev(xnew, tck, der=0)
            interp_mirror[i,:] = ynew

        self.mirror_stage_positions = mirror_stage_pos
        self.mirror_PCFS_interferogram = interp_mirror # not including the first line of position

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(interp_stage_pos)
        path_length_difference = 0.2 * (interp_stage_pos) # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # zeta_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram is by definition entirely symmetric.
        spectral_correlation = self.mirror_PCFS_interferogram.copy()
        for i in range(a):
            spectral_correlation[i,:] = np.abs(np.fft.fftshift(np.fft.fft(self.mirror_PCFS_interferogram[i,:])))

        self.splev_spec_corr = {}
        self.splev_spec_corr['spectral_corr'] = spectral_correlation
        self.splev_spec_corr['zeta'] = zeta_eV

    def plot_splev_spec_corr(self, tau_select, xlim):
        x = self.splev_spec_corr['zeta']
        ind = np.array([np.argmin(np.abs(self.tau - tau)) for tau in tau_select])
        legends = [tau/1e9 for tau in tau_select]
        y = self.splev_spec_corr['spectral_corr'][ind,:]

        plt.figure()
        plt.subplot(2,1,1)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:])

        plt.ylabel(r'$p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)
        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')

        plt.subplot(2,1,2)
        for i in range(len(tau_select)):
            plt.plot(x, y[i,:]/max(y[i,:]))

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')
        plt.xlim(xlim)
        plt.legend(legends)

        plt.title(self.PCFS_ID + r' Mirrored Spectral Correlation at $\tau$ [ms]')
        plt.tight_layout()
        plt.show()

    '''
    This function calculates and plots the spectral correlation of an interterferogram parsed as two vectors containing the stage_positions (not path length differences!), the corresponding interferogram values and the white-fringe position.
    '''
    def plot_spectral_corr(self, stage_positions, interferogram, white_fringe_pos):

        #some constants
        eV2cm = 8065.54429
        cm2eV = 1 / eV2cm

        N = len(stage_positions)
        path_length_difference = 2 * (stage_positions - white_fringe_pos) * 0.1 # NOTE: This is where we convert to path length difference space in cm.
        delta = (max(path_length_difference) - min(path_length_difference)) / (N-1)
        zeta_eV = np.fft.fftshift(np.fft.fftfreq(N, delta)) * cm2eV * 1000 # in meV

        # get reciprocal space (wavenumbers).
        # increment = 1 / delta
        # zeta_eV = np.linspace(-0.5 * increment, 0.5 * increment, num = N) * cm2eV * 1000 # converted to meV

        # take the FFT of the interferogram to get the spectral correlation. All that shifting is to shift the zero frequency component to the middle of the FFT vector. We take the real part of the FFT because the interferogram has a complex output and we are interested in the norm.
        spectral_correlation = np.abs(np.fft.fftshift(np.fft.fft(interferogram)))
  
       
        normalized_spectral_correlation = spectral_correlation / np.max(spectral_correlation, axis=1, keepdims=True)

      
        plt.plot(zeta_eV, normalized_spectral_correlation, '-o', markersize = 1)

        plt.ylabel(r'Normalized $p(\zeta)$')
        plt.xlabel(r'$\zeta$ [meV]')

        plt.title(self.PCFS_ID + r' Spectral Correlation at $\tau$ [ms]')
        plt.show()



    '''
    This function gets the fourier spectrum from the photon stream.
    '''
    def get_Fourier_spectrum_from_stream(self, bin_width, file_in):
        t = np.zeros(len(self.stage_positions)) # for intensity
        Fourier = np.zeros(len(self.stage_positions))
        self.get_file_photons()

        for f in self.file_photons:
            # looking to get cross correlation
            if 'sum' not in f:
                if file_in in f:
                    correlation_number = int(re.findall(r'\d+', f)[0]) # extract the number of correlation measurements from the file names
                    self.photons[f].get_intensity_trace(f, bin_width)
                    intensity = self.photons[f].intensity_counts['Trace']
                
                    t[correlation_number] = (np.sum(intensity[:,0]) + np.sum(intensity[:,1]))
                    Fourier[correlation_number] = (np.sum(intensity[:,0]) - np.sum(intensity[:,1])) / t[correlation_number]

        out_dic = {}
        out_dic['Fourier'] = Fourier
        out_dic['stage_positions'] = self.stage_positions
        out_dic['intensity'] = t
        self.Fourier[file_in] = out_dic



        #%%
#below is an example of how you would analyze data from a PCFS simulation
pcfs1 = PCFS(r"C:\Data and Code\Data\24_02_11_ensamble_PCFS_SIMS\Esize1", 2,simulation = True)

pcfs1.get_photons_all()
#%%

pcfs1.get_sum_signal_all()


#%%
pcfs1.get_intensity_correlations((1e1,1e11), 3)

#%%
pcfs1.get_blinking_corrected_PCFS()
x = []
for i in range(400):
    x.append(.1*i*10*1e6)

pcfs1.plot_spectral_diffusion([1e6,1e9], -4)

#%%

pcfs1.get_mirror_spectral_corr(0, 0)
pcfs1.get_splev_mirror_spec_corr( 0, 0)
#%%
pcfs1.plot_mirror_spectral_corr([1e7,1e9], (-10,10))
#%%

pcfs1.plot_splev_spec_corr([1e7,1e9], (-1,1))


#%%
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))
def lorenzian(x, x0, gamma, A):
    return A * (gamma / ((x - x0) ** 2 + gamma ** 2) / np.pi)

fwhm_arr = []
for i in range(len(pcfs1.tau)):
    y = pcfs1.mirror_spectral_correlation['spectral_corr'][i,:]/max(pcfs1.mirror_spectral_correlation['spectral_corr'][i,:])
    x = pcfs1.mirror_spectral_correlation['zeta']
    try:
        params, covariance = curve_fit(lorenzian, x, y)
    except RuntimeError:
        fwhm_arr.append(0)
    else:
        #for lorenzian
        fwhm = 2 * params[1] 
        #for gaussian
        #fwhm = 2 * np.sqrt(2 * np.log(2)) * params[2]
        fwhm_arr.append(fwhm)
        plt.plot(x, y, 'bo', label='Data')
        plt.plot(x, lorenzian(x, *params), 'r-', label='Fit')
        plt.legend()
        plt.show()

#%%
plt.scatter(pcfs1.tau,fwhm_arr, label='Scatter Plot on Log Scales')
plt.xscale('log')
plt.xlim(1e4,1e11)
plt.ylim(.5,2)
plt.show()
file_path = r'C:\Data and Code\Data\24_02_11_ensamble_PCFS_SIMS\FWHMensambleSize1.npy'

# Save the array to the specified file
np.save(file_path, fwhm_arr)

#%%


# Create a scatter plot with logarithmic scales and open circles
plt.figure(figsize=(10, 6), dpi=600)  # Adjust the figure size and DPI as needed

scatter = plt.scatter(pcfs1.tau, fwhm_arr, c='blue', marker='o', edgecolors='none', label='FWHM vs. Tau')
plt.xscale('log')
plt.xlim(1e4, 1e11)
plt.ylim(0.5, 3)

# Labeling and titles with larger font sizes
plt.xlabel('Tau', fontsize=16,  color='black')
plt.ylabel(rf'FWHM of $P(\zeta)$', fontsize=16, color='black')
plt.title('Spectral Diffusion', fontsize=18, fontweight='bold',color='black')

# Remove gridlines
plt.grid(False)

# Customize ticks and tick labels
plt.tick_params(axis='both', which='major', labelsize=14)
plt.tick_params(axis='both', which='minor', labelsize=12)

# Legend
plt.legend()

# Increase the marker size
scatter.set_sizes([50])
custom_dash = [10, 7]
# Connect the dots with very thin lines
plt.plot(pcfs1.tau, fwhm_arr, linestyle=(0, (custom_dash[0], custom_dash[1])), color='black', linewidth=0.5)
plt.savefig(r"C:\Data and Code\Data\23_10_06_4ATPsimulations\Plots\my_plot.png", dpi=300)
plt.show()





