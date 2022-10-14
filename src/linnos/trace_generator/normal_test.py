import numpy as np

KB = 1024
AVG_SIZE_BYTES = 1024*KB
STDDEV_BYTES =  256*KB
mu, sigma = AVG_SIZE_BYTES, STDDEV_BYTES

s = np.random.normal(AVG_SIZE_BYTES, STDDEV_BYTES, 100)

print(f"min IO size: {min(s)/1024} KB")

import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')

plt.savefig("normal.pdf")