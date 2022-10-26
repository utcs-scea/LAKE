import matplotlib.pyplot as plt
  

a = ["10", "100", "1000", "2000", "5000", "10000"]


_4096_dev_0 = [ 2281, 425.5, 403.446, 367.968, 200.479, 258.53]
_4096_dev_1 = [ 443, 597, 538.305, 403.987, 233.943, 338.014]
_16384_dev_0 = [ 296, 499.3, 499.181, 412.325, 305.199, 293.109]
_16384_dev_1 = [ 614, 610.8, 562.904, 418.386, 374.339, 399.167]
_65536_dev_0 = [ 872, 635.7, 507.542, 578.587, 538, 365.406]
_65536_dev_1 = [ 788, 698, 759.8, 595.029, 577.806, 617.141]
_262144_dev_0 = [ 1306, 820.4, 896.963, 887.187, 672.861, 768.583]
_262144_dev_1 = [ 1865, 1194, 979.78, 1058.13, 880.438, 897.564]
_1048576_dev_0 = [ 4429, 1848.2, 1410.18, 2064.27, 1651.7, 1840.06]
_1048576_dev_1 = [ 3661, 2843.9, 1955.02, 2138.36, 2198.67, 1973.9]
_4096_dev_0e = [ 0, 131.731, 157.525, 140.314, 83.9405, 108.379]
_4096_dev_1e = [ 0, 95.6609, 148.637, 126.448, 92.45, 127.953]
_16384_dev_0e = [ 0, 149.833, 149.402, 149.416, 136.836, 123.418]
_16384_dev_1e = [ 0, 182.28, 161.632, 164.961, 138.849, 139.474]
_65536_dev_0e = [ 0, 204.707, 192.138, 202.798, 198.363, 184.478]
_65536_dev_1e = [ 0, 242.279, 525.405, 205.961, 206.381, 191.632]
_262144_dev_0e = [ 0, 344.655, 382.183, 344.801, 356.108, 363.898]
_262144_dev_1e = [ 0, 240.722, 304.936, 314.827, 306.558, 336.068]
_1048576_dev_0e = [ 0, 952.13, 806.775, 844.36, 930.575, 989.08]
_1048576_dev_1e = [ 0, 789.089, 895.392, 811.422, 1091.44, 873.208]

plt.scatter(a, _4096_dev_0)
plt.errorbar(a, _4096_dev_0, yerr=_4096_dev_0e, fmt="o", label="4k Drive 1")

plt.scatter(a, _4096_dev_1)
plt.errorbar(a, _4096_dev_1, yerr=_4096_dev_1e, fmt="o", label="4k Drive 2")

plt.scatter(a, _16384_dev_0)
plt.errorbar(a, _16384_dev_0, yerr=_16384_dev_0e, fmt="o", label="16k Drive 1")

plt.scatter(a, _16384_dev_1)
plt.errorbar(a, _16384_dev_1, yerr=_16384_dev_1e, fmt="o", label="16k Drive 2")

plt.scatter(a, _65536_dev_0)
plt.errorbar(a, _65536_dev_0, yerr=_65536_dev_0e, fmt="o", label="64k Drive 1")

plt.scatter(a, _65536_dev_1)
plt.errorbar(a, _65536_dev_1, yerr=_65536_dev_1e, fmt="o", label="64k Drive 2")

plt.scatter(a, _262144_dev_0)
plt.errorbar(a, _262144_dev_0, yerr=_262144_dev_0e, fmt="o", label="256k Drive 2")

plt.scatter(a, _262144_dev_1)
plt.errorbar(a, _262144_dev_1, yerr=_262144_dev_1e, fmt="o", label="256k Drive 2")

plt.scatter(a, _1048576_dev_0)
plt.errorbar(a, _1048576_dev_0, yerr=_1048576_dev_0e, fmt="o", label="1024k Drive 1")

plt.scatter(a, _1048576_dev_1)
plt.errorbar(a, _1048576_dev_1, yerr=_1048576_dev_1e, fmt="o", label="1024k Drive 2")

plt.ylabel('Avg Latency (us)')
plt.xlabel('IOPS')

plt.legend(loc='upper left')

plt.savefig(f"latency.png", bbox_inches='tight', pad_inches=0.05)
