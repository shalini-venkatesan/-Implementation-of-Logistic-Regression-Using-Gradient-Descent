# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.
. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SHALINI VENKATESAN
RegisterNumber:  212222240096
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j

def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```

## Output:

#### Array value of X:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/42045068-a2c4-42e4-8913-fcde2788bdbc)

#### Array value of Y:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/aff41d26-9c3d-4107-8531-8420da9fecae)

#### Exam 1-Score graph:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/03720f54-b391-4090-b488-3f5a7eca6fee)

#### Sigmoid function graph:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/4fa5f881-e348-410b-8d08-98b5c2bae583)

#### X_Train_grad value:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/da4e301d-0a1d-46d9-b731-f9c9f2dee214)

#### Y_Train_grad value:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/061c8864-4e34-4ceb-aced-f287af142435)

#### Print res.X:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/061c8864-4e34-4ceb-aced-f287af142435)

#### Decision boundary-gragh for exam score:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/df276009-64d1-4648-8142-18ac2e72fe08)

#### Probability value:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/d28eff8d-4445-4866-ac9e-78d3449d63ae)

#### Prediction value of mean:

![image](https://github.com/JoyceBeulah/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118343698/5ccd966f-e001-43fe-b6ab-7c25fc66934a)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

