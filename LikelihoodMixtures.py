import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

np.random.seed(110) # for reproducible random results

# set parameters
bins=20
sample_size = 20
red_mean = 3
red_std = 0.8

blue_mean = 7
blue_std = 2

# draw 20 samples from normal distributions with red/blue parameters

red = np.random.normal(red_mean, red_std, size=sample_size)
blue = np.random.normal(blue_mean, blue_std, size=sample_size)

both_colours = np.sort(np.concatenate((red, blue)))

#Best Fit
mu_red,std_red = sts.norm.fit(red)
mu_blue,std_blue = sts.norm.fit(blue)

#Histogram Generation
x_red=np.arange(mu_red-3*std_red,mu_red+3*std_red,0.1)
x_blue=np.arange(mu_blue-3*std_blue,mu_blue+3*std_blue,0.5)
y_red = sts.norm.pdf(x_red,mu_red,std_red)*sample_size
y_blue = sts.norm.pdf(x_blue,mu_blue,std_blue)*sample_size
plt.plot(x_red, y_red, 'r--', linewidth=1)
plt.plot(x_blue, y_blue, 'b--', linewidth=1)
plt.hist(red,bins,alpha=0.5,label='Class 1',color='red')
plt.hist(blue,bins,alpha=0.5,label='Class 2',color='blue')
plt.title('Gaussian Mixture (Specified)',fontweight="bold")
plt.legend()
plt.show()
