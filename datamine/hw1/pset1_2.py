import numpy,pandas,matplotlib
import numpy as np
import matplotlib.pyplot as plt

"""
problem 2:
Let Xi be an iid Bernoulli random variable with value {-1,1}.
Look at the random variable Zn = 1/n SUM(X_i).
By taking 1000 draws from Zn, plot its histogram.
Check that for small n (say, 5-10) Zn does not look that much like a Gaussian,
 but when n is bigger (already by the time n = 30 or 50) it looks much more like a Gaussian.
 Check also for much bigger n: n = 250, to see that at this point, one can really see the bell curve.
"""

# TODO: not sure ...
def bernoulli(**kwargs):
  samplesize=kwargs["samplesize"]
  #  print(np.random.seed)
  print("bernoulli with size: " + str(samplesize))

  # By taking 1000 draws from Zn, plot its histogram.
  draws = 1000
  rv_z_list = []  # put the draws in here
  for i in range(0,draws):
    # random.choice([-1,1]): Let Xi be an iid Bernoulli random variable with value {-1,1}.
    # .mean():               Look at the random variable Zn = 1/n SUM(X_i).
    rv = np.random.choice([-1,1],size=samplesize).mean()
    rv_z_list.append(rv)
  # verify this is 1000
  print("-D:" + str(len(rv_z_list)) + " == " + str(draws) + "?")
  #import ipdb;ipdb.set_trace()

  # plot
  count, bins, ignored = plt.hist(rv_z_list, bins=30, normed=True)
  plt.show()
# end bernoulli


seed=23
np.random.seed(seed)

# Check that for small n (say, 5-10) Zn does not look that much like a Gaussian,
size = 10
bernoulli(samplesize=size)
#  but when n is bigger (already by the time n = 30 or 50) it looks much more like a Gaussian.
size = 30
bernoulli(samplesize=size)
#  Check also for much bigger n: n = 250, to see that at this point, one can really see the bell curve.
size = 250
bernoulli(samplesize=size)
size = 1000
bernoulli(samplesize=size)

# rejects

"""
# check that bernoulli is 0.5
p = 0.5
# call binomial
s1 = np.random.binomial(draws,p,size)
print(s1)
##############################
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randint.html
# bad, includes 0!
bernoulli_rv_list = numpy.random.randint(-1,1,samplesize)
print(bernoulli_rv_list)
# mean
bernoulli_rv_list.mean()
##############################
##############################
"""
