import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import timeit
from joblib import Parallel,delayed
import cv2
import numpy as np

noise_chunk = 'chunk1'

noise_csv_path = f'{noise_chunk}/{noise_chunk}.csv'
noise_sig_path = f'{noise_chunk}/{noise_chunk}.hdf5'

noise = pd.read_csv(noise_csv_path)

chunk_name = noise 
noisepath = noise_sig_path 

img_save_path = 'images/NO/'

eqlist = chunk_name['trace_name'].to_list()
print(len(eqlist))

dtfl = h5py.File(noisepath, 'r')

step = 15
total_data = 10000

for_loop_total = total_data * step 
obtained_traces = []

count_data = 0
for n in range(0, for_loop_total, step):
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
    count_data+=1
    print(count_data, eqlist[n])

np.save(img_save_path+eq_chunk+'_obtained_traces.npy', np.array(obtained_traces))
a = np.load(img_save_path+eq_chunk+'_obtained_traces.npy')

print(a)
    
    
