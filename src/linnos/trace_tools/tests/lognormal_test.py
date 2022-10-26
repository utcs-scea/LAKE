import numpy as np
import matplotlib.pyplot as plt
import math

KB = 1024
MB = KB * 1024
AVG_SIZE_BYTES = 430*KB
MAX_BYTES = 9 * MB
STDDEV_BYTES =  (math.log(MAX_BYTES) - math.log(AVG_SIZE_BYTES))/3
print(STDDEV_BYTES)

s = np.random.lognormal(math.log(AVG_SIZE_BYTES), STDDEV_BYTES, 10000)
s[s > MAX_BYTES] = MAX_BYTES
# for sample in s:
#     if sample > MAX_BYTES:
#         sample = MAX_BYTES
print(np.mean(s))
print(np.max(s))
print(np.min(s))
mu, sigma = math.log(AVG_SIZE_BYTES), STDDEV_BYTES
count, bins, ignored = plt.hist(s, 100, density=True, align='mid')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
       / (x * sigma * np.sqrt(2 * np.pi)))

plt.plot(x, pdf, linewidth=2, color='r')
plt.axis('tight')

plt.savefig("lognormal.pdf")

