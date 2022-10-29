# Part of LAKE: Towards a Machine Learning-Assisted Kernel with LAKE
# Copyright (C) 2022-2024 Henrique Fingler
# Copyright (C) 2022-2024 Isha Tarte
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import matplotlib.pyplot as plt
  

a = ["100", "1000", "5000", "10000", "20000", "50000"]


_4096_dev_0 = [ 68.83, 65.3663, 64.9471, 65.6281, 65.7488, 65.9095]
_4096_dev_1 = [ 74.93, 71.7737, 72.0261, 72.8689, 72.5683, 72.3827]
_4096_dev_2 = [ 75.06, 71.1695, 72.0716, 71.297, 71.43, 71.6851]
_16384_dev_0 = [ 107.83, 108.217, 104.948, 105.329, 105.259, 105.685]
_16384_dev_1 = [ 117.94, 118.858, 117.413, 118.277, 118.138, 118.567]
_16384_dev_2 = [ 120.88, 118.199, 117.391, 117.559, 117.516, 117.375]
_65536_dev_0 = [ 127.21, 124.818, 124.633, 125.425, 125.316, 124.783]
_65536_dev_1 = [ 138.25, 135.946, 135.321, 135.648, 135.622, 135.446]
_65536_dev_2 = [ 136.34, 135.168, 134.739, 134.311, 134.706, 134.302]
_262144_dev_0 = [ 184.21, 179.285, 179.83, 177.49, 178.646, 177.312]
_262144_dev_1 = [ 192.09, 189.057, 188.411, 188.625, 188.418, 188.446]
_262144_dev_2 = [ 177.93, 185.793, 185.076, 185.532, 184.117, 185.62]
_1048576_dev_0 = [ 403.51, 392.478, 392.165, 391.34, 396.553, 390.505]
_1048576_dev_1 = [ 403.57, 405.814, 403.47, 405.085, 404.64, 405.501]
_1048576_dev_2 = [ 417.09, 416.505, 412.577, 403.819, 404.73, 405.827]
_4096_dev_0e = [ 20.2978, 17.6068, 2, 1, 0, 0]
_4096_dev_1e = [ 14.1067, 12.3288, 1, 0, 0, 1.41421]
_4096_dev_2e = [ 13.784, 13, 0, 0, 1, 0]
_16384_dev_0e = [ 43.2204, 33.9853, 26.1151, 3, 1.73205, 2]
_16384_dev_1e = [ 28.775, 23.3452, 16.9411, 16.8819, 17.4642, 16.8523]
_16384_dev_2e = [ 24.5357, 21.3307, 19.1833, 17.3205, 18.5203, 15.4272]
_65536_dev_0e = [ 34.4674, 31.5911, 20.8327, 10.198, 6.32456, 5.91608]
_65536_dev_1e = [ 23.5584, 17.6068, 9.48683, 6.9282, 6.55744, 6.40312]
_65536_dev_2e = [ 26.6646, 18.1108, 9.53939, 7.07107, 7.28011, 7]
_262144_dev_0e = [ 37.55, 30.6105, 19.1833, 16.1555, 15.6844, 16.2788]
_262144_dev_1e = [ 27.6767, 21.166, 15.4272, 12.8452, 13.8924, 13.3791]
_262144_dev_2e = [ 24.2899, 22.0907, 14.2478, 12.4499, 12.4499, 13.4536]
_1048576_dev_0e = [ 56.8946, 33.6006, 28.7576, 27.1293, 31.1609, 27.1662]
_1048576_dev_1e = [ 47.3075, 28.2489, 24.0208, 25.5734, 23.388, 24.7992]
_1048576_dev_2e = [ 45.3211, 36.3593, 31.749, 24.8596, 23.1084, 26.3439]

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
