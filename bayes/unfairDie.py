import numpy as np
from numpy.random import choice

#num_trials = 100000
num_trials = 10000
mean_sum = 0
# YOUR CODE HERE


feedback = int(np.round(num_trials/10))
#let's get a die rolling randomly

for t in range(1, num_trials+1):

  if(t%feedback==0):
    print(np.round(100*t/num_trials,3), '% complete: mean_sum=', mean_sum/t)
  #roll a fair six-sided die
  roll1 = (np.random.choice(6,1))+1
  roll2 = (np.random.choice(6,1,p=[0.1,0.1,0.1,0.1,0.1,0.5]))+1
  mean_sum = mean_sum + roll1+roll2
  #print("Roll Fair = " , roll1)
  #print("Roll UnFair = " , roll2)


print('mean sum =', mean_sum/num_trials)