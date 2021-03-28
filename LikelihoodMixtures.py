import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt

def estimate_mean(data, weight):
    return np.sum(data * weight) / np.sum(weight)

def estimate_std(data, weight, mean):
    variance = np.sum(weight * (data - mean)**2) / np.sum(weight)
    return np.sqrt(variance)

#np.random.seed(110) # for reproducible random results

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

#Using sts functions only 

# =============================================================================
# #Best Fit
# mu_red,std_red = sts.norm.fit(red)
# mu_blue,std_blue = sts.norm.fit(blue)
# =============================================================================

# =============================================================================
# #Show Fitted Parameters
# print("\nRed Mean:",mu_red,", Red Std:",std_red)
# print("\nRed Real Mean:",red_mean,", Red Real Std:",red_std)
# 
# print("\nBlue Mean:",mu_blue,", Blue Std:",std_blue)
# print("\nBlue Real Mean:",blue_mean,", Blue Real Std:",blue_std)
# =============================================================================

#Histogram Generation
x_red=np.arange(red_mean-3*red_std,red_mean+3*red_std,0.1)
x_blue=np.arange(blue_mean-3*blue_std,blue_mean+3*blue_std,0.5)
y_red = sts.norm.pdf(x_red,red_mean,red_std)*sample_size
y_blue = sts.norm.pdf(x_blue,blue_mean,blue_std)*sample_size
plt.plot(x_red, y_red, 'r-', linewidth=1)
plt.plot(x_blue, y_blue, 'b-', linewidth=1)
plt.hist(red,bins,alpha=0.5,label='Class 1',color='red')
plt.hist(blue,bins,alpha=0.5,label='Class 2',color='blue')
plt.title('Gaussian Mixture (Specified)',fontweight="bold")
plt.legend()
plt.show()

#Real Situation Histogram
plt.plot(x_red, y_red, 'r-', linewidth=1,label='Exact Red')
plt.plot(x_blue, y_blue, 'b-', linewidth=1,label='Exact Blue')
plt.hist(red,bins,alpha=0.5,color='purple')
plt.hist(blue,bins,alpha=0.5,color='purple')
plt.title('Gaussian Mixture (Unspecified)',fontweight="bold")
plt.legend()
plt.show()

#======================= Bayesian Process =======================

#First Estimates
red_mean_guess = 1.1 
blue_mean_guess = 9 
# estimates for the standard deviation 
red_std_guess = 2 
blue_std_guess = 1.7
#Gaussian likelihoods and Likelihood Ratio
likelihood_of_red = sts.norm(red_mean_guess, red_std_guess).pdf(both_colours)
likelihood_of_blue = sts.norm(blue_mean_guess, blue_std_guess).pdf(both_colours)
likelihood_total = likelihood_of_red + likelihood_of_blue
red_weight = likelihood_of_red / likelihood_total
blue_weight = likelihood_of_blue / likelihood_total

# new estimates for standard deviation
blue_std_guess = estimate_std(both_colours, blue_weight, blue_mean_guess)
red_std_guess = estimate_std(both_colours, red_weight, red_mean_guess)
# new estimates for mean
red_mean_guess = estimate_mean(both_colours, red_weight)
blue_mean_guess = estimate_mean(both_colours, blue_weight)

#================ Real Situation Histogram Fit ================

#Exact curves
plt.plot(x_red, y_red, 'r-', linewidth=1,label='Exact Red')
plt.plot(x_blue, y_blue, 'b-', linewidth=1,label='Exact Blue')

#Repeat many times the estimation process (Very ugly to see)
for i in range(0,5):
    likelihood_of_red = sts.norm(red_mean_guess, red_std_guess).pdf(both_colours)
    likelihood_of_blue = sts.norm(blue_mean_guess, blue_std_guess).pdf(both_colours)
    likelihood_total = likelihood_of_red + likelihood_of_blue
    red_weight = likelihood_of_red / likelihood_total
    blue_weight = likelihood_of_blue / likelihood_total
    # new estimates for standard deviation
    blue_std_guess = estimate_std(both_colours, blue_weight, blue_mean_guess)
    red_std_guess = estimate_std(both_colours, red_weight, red_mean_guess)
    # new estimates for mean
    red_mean_guess = estimate_mean(both_colours, red_weight)
    blue_mean_guess = estimate_mean(both_colours, blue_weight)

#Fitted curves
x_red_bayes=np.arange(red_mean_guess-3*red_std_guess,
                      red_mean_guess+3*red_std_guess,0.1)
x_blue_bayes=np.arange(blue_mean_guess-3*blue_std_guess,
                       blue_mean_guess+3*blue_std_guess,0.5)
y_red_bayes = sts.norm.pdf(x_red_bayes,red_mean_guess,
                           red_std_guess)*sample_size
y_blue_bayes = sts.norm.pdf(x_blue_bayes,blue_mean_guess,
                            blue_std_guess)*sample_size
plt.plot(x_red_bayes, y_red_bayes, 'r--', linewidth=1,
         label='Red Fit')
plt.plot(x_blue_bayes, y_blue_bayes, 'b--', linewidth=1,
         label='Blue Fit')
plt.hist(red,bins,alpha=0.5,color='purple')
plt.hist(blue,bins,alpha=0.5,color='purple')
plt.title('Gaussian Mixture with Fit (5 iterations)',fontweight="bold")
plt.legend()
plt.show()
