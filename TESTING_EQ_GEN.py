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
print(len(eqlist))
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
for n in range(0, for_loop_total, step):
    if n in [193,194,195,196,197]:
        continue
    dataset = dtfl.get('data/'+str(eqlist[n]))
    data = np.array(dataset)

    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(img_save_path+'mag/'+eqlist[n]+'.png',bbox_inches='tight',dpi=50)
    plt.close()

    fig, ax = plt.subplots(figsize=(3,2))
    ax.specgram(data[:,2],Fs=100,NFFT=256,cmap="jet",vmin=-10,vmax=25)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.savefig(img_save_path+'freq/'+eqlist[n]+'.png',bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
    plt.close()

    obtained_traces.append(eqlist[n])
    obtained_magnitude.append(magnitude[n])
    obtained_distance.append(distance[n])

    count_data+=1
    print(count_data, eqlist[n])

np.save(img_save_path+eq_chunk+'_obtained_traces.npy', np.array(obtained_traces))
np.save(img_save_path+eq_chunk+'_obtained_magnitude.npy', np.array(obtained_magnitude))
np.save(img_save_path+eq_chunk+'_obtained_distance.npy', np.array(obtained_distance))

a = np.load(img_save_path+eq_chunk+'_obtained_traces.npy')
b = np.load(img_save_path+eq_chunk+'_obtained_magnitude.npy')
c = np.load(img_save_path+eq_chunk+'_obtained_distance.npy')
print(a, b, c)
    
    
