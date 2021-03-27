import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts

#Sample structure: X=(x_1,x_2)
X_1=np.array([[4,2],[2,4],[2,3],[3,6],[4,4]])
X_2=np.array([[9,10],[6,8],[9,5],[8,7],[10,8]])

mean_1=np.sum(X_1,axis=0)/5
mean_2=np.sum(X_2,axis=0)/5

cov_1=np.zeros((2,2))
cov_2=np.zeros((2,2))

for i in range(0,5):
    cov_1+=np.outer(X_1[i]-mean_1,X_1[i]-mean_1)/4

for i in range(0,5):
    cov_2+=np.outer(X_2[i]-mean_2,X_2[i]-mean_2)/4
    
with_scatt=cov_1+cov_2
betw_scatt=np.outer(mean_1-mean_2,mean_1-mean_2)

#Eigenvalue problem
A=np.dot(np.linalg.inv(with_scatt),betw_scatt)
w,v=np.linalg.eig(A)

#Direction of projection
min_eig_line = np.poly1d([v[1,1]/v[0,1],0])
max_eig_line = np.poly1d([v[1,0]/v[0,0],0])
x_range_min = np.arange(-7,1,1)
x_range_max = np.arange(0,11.5,1)

#Points plot with vector relative to the max eigenvalue
plt.plot(X_1.T[0],X_1.T[1],'r.',label='Class 1')
plt.plot(X_2.T[0],X_2.T[1],'b.',label='Class 2')
plt.plot(x_range_max,max_eig_line(x_range_max),'gold',
         label='Eigenvector')
plt.title('Projection Line, Max Eigenvalue',fontweight="bold")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Points plot with vector relative to the min eigenvalue
plt.plot(X_1.T[0],X_1.T[1],'r.',label='Class 1')
plt.plot(X_2.T[0],X_2.T[1],'b.',label='Class 2')
plt.plot(x_range_min,min_eig_line(x_range_min),'gold',
         label='Eigenvector')
plt.title('Projection Line, Min Eigenvalue',fontweight="bold")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#New points after a rotation operator application

#Min eigenvalue
theta_min = np.arctan(v[1,1]/v[0,1])
theta_max = np.arctan(v[1,0]/v[0,0])
rot_min = np.array([[np.cos(theta_min), -np.sin(theta_min)], 
                    [np.sin(theta_min),  np.cos(theta_min)]])

rot_max = np.array([[np.cos(theta_max), -np.sin(theta_max)], 
                    [np.sin(theta_max),  np.cos(theta_max)]])

new_X_1=np.dot(X_1,rot_min)
new_X_2=np.dot(X_2,rot_min)
plt.plot(new_X_1.T[0],new_X_1.T[1],'r.',label='Class 1')
plt.plot(new_X_2.T[0],new_X_2.T[1],'b.',label='Class 2')
plt.title('Rotated Plane, Min Eigenvalue',fontweight="bold")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Max eigenvalue
New_X_1=np.dot(X_1,rot_max)
New_X_2=np.dot(X_2,rot_max)
plt.plot(New_X_1.T[0],New_X_1.T[1],'r.',label='Class 1')
plt.plot(New_X_2.T[0],New_X_2.T[1],'b.',label='Class 2')
plt.title('Rotated Plane, Max Eigenvalue',fontweight="bold")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#Distribution of the projections

#Best Fit
mu_red,std_red = sts.norm.fit(new_X_1.T[0])
mu_blue,std_blue = sts.norm.fit(new_X_2.T[0])

#Min Eigenvalue Projection
x_red=np.arange(mu_red-4*std_red,mu_red+4*std_red,0.1)
x_blue=np.arange(mu_blue-4*std_blue,mu_blue+4*std_blue,0.1)
y_red = sts.norm.pdf(x_red,mu_red,std_red)*5
y_blue = sts.norm.pdf(x_blue,mu_blue,std_blue)*5
plt.plot(x_red, y_red, 'r--', linewidth=1)
plt.plot(x_blue, y_blue, 'b--', linewidth=1)

plt.hist(new_X_1.T[0],20,alpha=0.5,label='Class 1',color='red')
plt.hist(new_X_2.T[0],20,alpha=0.5,label='Class 2',color='blue')
plt.title('Projection Distribution, Min Eig',fontweight="bold")
plt.legend()
plt.show()

#Best Fit
mu_red,std_red = sts.norm.fit(New_X_1.T[0])
mu_blue,std_blue = sts.norm.fit(New_X_2.T[0])

#Max Eigenvalue Projection
x_red=np.arange(mu_red-4*std_red,mu_red+4*std_red,0.1)
x_blue=np.arange(mu_blue-4*std_blue,mu_blue+4*std_blue,0.1)
y_red = sts.norm.pdf(x_red,mu_red,std_red)*5
y_blue = sts.norm.pdf(x_blue,mu_blue,std_blue)*5
plt.plot(x_red, y_red, 'r--', linewidth=1)
plt.plot(x_blue, y_blue, 'b--', linewidth=1)

plt.hist(New_X_1.T[0],20,alpha=0.5,label='Class 1',color='red')
plt.hist(New_X_2.T[0],15,alpha=0.5,label='Class 2',color='blue')
plt.title('Projection Distribution, Max Eig',fontweight="bold")
plt.legend()
plt.show()

