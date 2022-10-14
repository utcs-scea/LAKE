import numpy as np


a = 4.0
n = 20000
s = np.random.default_rng().zipf(a, size=n)

print(s[:40])

import matplotlib.pyplot as plt
from scipy.special import zeta  

count = np.bincount(s)
k = np.arange(1, s.max() + 1)

plt.bar(k, count[1:], alpha=0.5, label='sample count')
plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5,
         label='expected count')  
plt.semilogy()
plt.grid(alpha=0.4)
plt.legend()
plt.title(f'Zipf sample, a={a}, size={n}')
plt.savefig("zipf.pdf")