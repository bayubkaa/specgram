import numpy as np

import matplotlib.pyplot as plt



x = np.load('testing/EQ/chunk2_obtained_distance.npy')
print(x)
num_bins = 50
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)

# add a 'best fit' line

plt.xlabel('Distance')
#plt.ylabel('Frequency')
plt.title(r'Earthquake Distance Distribution')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.savefig('Distance_testing.png')
plt.show()