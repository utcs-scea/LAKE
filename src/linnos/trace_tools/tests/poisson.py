import numpy as np



ARRIVAL_RATE_US = 70

s= np.random.exponential(ARRIVAL_RATE_US, 2000)

print(f"min: {min(s)}")
print(f"min: {max(s)}")

import matplotlib.pyplot as plt


bin = np.arange(0,ARRIVAL_RATE_US*10,1)

plt.hist(s, bins=bin, edgecolor='blue') 
plt.title("Exponential Distribution") 
plt.savefig("poisson.pdf")
