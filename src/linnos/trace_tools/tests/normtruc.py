import matplotlib.pyplot as plt
import scipy.stats as stats
import statistics

lower, upper = 12*1000, 64*1000
mu, sigma = 25*1000, 12*1000
X = stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
N = stats.norm(loc=mu, scale=sigma)

fig, ax = plt.subplots(2, sharex=True)
ax[0].hist(X.rvs(10000))
ax[1].hist(N.rvs(10000))
plt.savefig("normtrunc.pdf")


print(f"min {min(X.rvs(1000))}")
print(f"max {max(X.rvs(1000))}")
print(f"mean {statistics.mean(X.rvs(1000))}")