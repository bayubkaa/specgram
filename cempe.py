import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from joblib import Parallel,delayed
import cv2
import numpy as np

eq_chunk = 'chunk2'

eq_csv_path = f'{eq_chunk}/{eq_chunk}.csv'
eq_sig_path = f'{eq_chunk}/{eq_chunk}.hdf5'

earthquakes = pd.read_csv(eq_csv_path)

earthquakes = earthquakes[(earthquakes.trace_category == 'earthquake_local') & 
                            ((earthquakes.source_distance_km >= 0) & (earthquakes.source_distance_km <= 150)) & 
                            ((earthquakes.source_magnitude >= 0) & (earthquakes.source_magnitude <= 5))]

chunk_name = earthquakes 
eqpath = eq_sig_path 

img_save_path = 'testing/EQ/'

eqlist = chunk_name['trace_name'].to_list()

magnitude = chunk_name['source_magnitude'].to_list()
distance = chunk_name['source_distance_km'].to_list()

dtfl = h5py.File(eqpath, 'r')

step = 13
total_data = 10000

for_loop_total = total_data * step 

obtained_traces = []
obtained_magnitude = []
obtained_distance = []

count_data = 0
for n in range(25, for_loop_total, step):
    if n in [193,194,195,196,197]:
        continue
    dataset = dtfl.get('data/'+str(eqlist[n]))
    data = np.array(dataset)

    # fig, ax = plt.subplots(figsize=(3,2))
    # ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
    # ax.set_xlim([0,60])
    # ax.axis('off')
    # plt.gca().set_axis_off()
    # plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
    #             hspace = 0, wspace = 0)
    # plt.margins(0,0)
    # plt.savefig(img_save_path+'mag/'+eqlist[n]+'.png',bbox_inches='tight',dpi=50)
    # plt.close()
    #data[:,2][:sampling_size]
    #data_dummy = np.array([2,1,-1,5,0,3,0,-4])


    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.margins(x=0)
    ax1.plot(data[:,2])
    Pxx, freqs, bins, im = ax2.specgram(data[:,2], NFFT=256, Fs=100, cmap="jet",vmin=-10,vmax=25)
    # The `specgram` method returns 4 objects. They are:
    # - Pxx: the periodogram
    # - freqs: the frequency vector
    # - bins: the centers of the time bins
    # - im: the .image.AxesImage instance representing the data in the plot

    xcoords = bins#[0.22058956, 0.33088437, 2.20589566]
    for xc in xcoords:
        ax2.axvline(x=xc)

    print(Pxx)
    print(freqs)
    print(bins)

    plt.show()
    
    
    break
    
    
  

    
    
    


