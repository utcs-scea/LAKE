import matplotlib.pyplot as plt
  

a = ["100", "1000", "5000", "10000", "20000", "50000"]


_4096_dev_0 = [ 155.88, 65.9189, 66.0324, 66.3294, 66.1215, 66.2005]
_4096_dev_1 = [ 159.45, 72.6811, 73.1991, 72.7836, 72.0321, 72.4178]
_4096_dev_2 = [ 160.06, 70.8747, 71.9033, 71.158, 72.0344, 71.8784]
_16384_dev_0 = [ 106.09, 104.976, 107.329, 106.917, 106.108, 106.384]
_16384_dev_1 = [ 121.55, 119.159, 118.234, 118.333, 117.762, 118.409]
_16384_dev_2 = [ 123.09, 117.826, 117.839, 117.349, 118.183, 117.937]
_65536_dev_0 = [ 130.43, 126.807, 124.96, 126, 125.593, 125.162]
_65536_dev_1 = [ 137.77, 135.533, 135.871, 135.61, 135.314, 135.465]
_65536_dev_2 = [ 137.78, 134.647, 134.613, 135.127, 134.789, 134.833]
_262144_dev_0 = [ 180.65, 176.399, 177.212, 178.251, 177.733, 178.875]
_262144_dev_1 = [ 189.51, 189.858, 188.9, 187, 187.793, 186.865]
_262144_dev_2 = [ 184.88, 186.068, 185.63, 185.672, 185.636, 186.148]
_1048576_dev_0 = [ 393.49, 390.797, 392.656, 392.658, 391.546, 391.765]
_1048576_dev_1 = [ 408.31, 406.696, 405.671, 403.721, 405.411, 405.158]
_1048576_dev_2 = [ 410.6, 403.224, 405.772, 412.523, 404.858, 410.384]
_4096_dev_0e = [ 850.579, 17.1172, 0, 1.41421, 0, 0]
_4096_dev_1e = [ 851.709, 11.8743, 1, 1, 2.82843, 0]
_4096_dev_2e = [ 851.543, 12.9615, 4.69042, 0, 0, 1]
_16384_dev_0e = [ 45.6727, 37.1484, 24.454, 2.82843, 0, 1.41421]
_16384_dev_1e = [ 21.1424, 22.383, 18.3848, 16.3095, 17.6068, 16.7929]
_16384_dev_2e = [ 28.6007, 24.9399, 18.0278, 17.9444, 17.8045, 11.3137]
_65536_dev_0e = [ 35.7351, 28.7576, 21.3073, 19.4679, 5.83095, 6.08276]
_65536_dev_1e = [ 24.4131, 19, 10, 7.81025, 7.4162, 7.54983]
_65536_dev_2e = [ 21.5639, 16.4924, 8.544, 6.78233, 7.14143, 7.74597]
_262144_dev_0e = [ 32.3574, 27.7669, 19.5448, 15.2315, 17.4929, 16.2173]
_262144_dev_1e = [ 23.0217, 22.7596, 14.7648, 12.3288, 12.7279, 12]
_262144_dev_2e = [ 27.8388, 21.7256, 15.1327, 12.5698, 13.4164, 12.3288]
_1048576_dev_0e = [ 44.9889, 33.9853, 27.7128, 28.4253, 26.6271, 27.8209]
_1048576_dev_1e = [ 45.913, 30.1496, 24.98, 25.2784, 24.5153, 23.9165]
_1048576_dev_2e = [ 50.1199, 30.4795, 27.0185, 29.4958, 27.0555, 28.2843]

plt.scatter(a, _4096_dev_0)
plt.errorbar(a, _4096_dev_0, yerr=_4096_dev_0e, fmt="o", label="4k Drive 1")

plt.scatter(a, _4096_dev_1)
plt.errorbar(a, _4096_dev_1, yerr=_4096_dev_1e, fmt="o", label="4k Drive 2")

plt.scatter(a, _4096_dev_2)
plt.errorbar(a, _4096_dev_2, yerr=_4096_dev_2e, fmt="o", label="4k Drive 3")


plt.scatter(a, _16384_dev_0)
plt.errorbar(a, _16384_dev_0, yerr=_16384_dev_0e, fmt="o", label="16k Drive 1")

plt.scatter(a, _16384_dev_1)
plt.errorbar(a, _16384_dev_1, yerr=_16384_dev_1e, fmt="o", label="16k Drive 2")

plt.scatter(a, _16384_dev_2)
plt.errorbar(a, _16384_dev_2, yerr=_16384_dev_2e, fmt="o", label="16k Drive 3")


plt.scatter(a, _65536_dev_0)
plt.errorbar(a, _65536_dev_0, yerr=_65536_dev_0e, fmt="o", label="64k Drive 1")

plt.scatter(a, _65536_dev_1)
plt.errorbar(a, _65536_dev_1, yerr=_65536_dev_1e, fmt="o", label="64k Drive 2")

plt.scatter(a, _65536_dev_2)
plt.errorbar(a, _65536_dev_2, yerr=_65536_dev_2e, fmt="o", label="64k Drive 3")



plt.scatter(a, _262144_dev_0)
plt.errorbar(a, _262144_dev_0, yerr=_262144_dev_0e, fmt="o", label="256k Drive 2")

plt.scatter(a, _262144_dev_1)
plt.errorbar(a, _262144_dev_1, yerr=_262144_dev_1e, fmt="o", label="256k Drive 2")

plt.scatter(a, _262144_dev_2)
plt.errorbar(a, _262144_dev_2, yerr=_262144_dev_2e, fmt="o", label="256k Drive 3")


plt.scatter(a, _1048576_dev_0)
plt.errorbar(a, _1048576_dev_0, yerr=_1048576_dev_0e, fmt="o", label="1024k Drive 1")

plt.scatter(a, _1048576_dev_1)
plt.errorbar(a, _1048576_dev_1, yerr=_1048576_dev_1e, fmt="o", label="1024k Drive 2")

plt.scatter(a, _1048576_dev_2)
plt.errorbar(a, _1048576_dev_2, yerr=_1048576_dev_2e, fmt="o", label="1024k Drive 3")


plt.ylabel('Avg Latency (us)')
plt.xlabel('IOPS')


ax = plt.subplot(111)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

#plt.legend(loc='upper left')

plt.savefig(f"latency.png", bbox_inches='tight', pad_inches=0.05)

"""
_4096_dev_0percent_late_io = [ 0, 0, 0, 0.00152905, 0.801314, 0.999322]
_4096_dev_1percent_late_io = [ 0, 0, 0, 0.00152905, 0.912141, 1]
_4096_dev_2percent_late_io = [ 0, 0, 0, 0, 0.908724, 1]
_16384_dev_0percent_late_io = [ 0, 0, 0, 0.666667, 0.832542, 1]
_16384_dev_1percent_late_io = [ 0, 0, 0, 0.760544, 0.958537, 1]
_16384_dev_2percent_late_io = [ 0, 0, 0, 0.72428, 0.949398, 1]
_65536_dev_0percent_late_io = [ 0, 0, 0.00252525, 0.845506, 0.921123, 1]
_65536_dev_1percent_late_io = [ 0, 0, 0.0227273, 0.936798, 0.995885, 1]
_65536_dev_2percent_late_io = [ 0, 0, 0.00757576, 0.935211, 0.995862, 1]
_262144_dev_0percent_late_io = [ 0, 0, 0.20122, 0.98913, 1, 1]
_262144_dev_1percent_late_io = [ 0, 0, 0.285714, 1, 1, 1]
_262144_dev_2percent_late_io = [ 0, 0, 0.244444, 1, 1, 1]
_1048576_dev_0percent_late_io = [ 0, 0, 1, 1, 1, 1]
_1048576_dev_1percent_late_io = [ 0, 0, 1, 1, 1, 1]

"""
