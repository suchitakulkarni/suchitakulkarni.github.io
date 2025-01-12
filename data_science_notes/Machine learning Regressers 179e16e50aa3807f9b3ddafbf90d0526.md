# Machine learning: Regressers

Created by: Suchita Kulkarni
Created time: January 12, 2025 8:24 AM
Tags: Machine learning notes, Regression

These notes focus on regression techniques and will be updated over time. Currently, they cover only pure regression techniques and exclude methods like random forests that can perform both regression and classification. 

# Regression techniques

Regression techniques are used when one want to predict a value as opposed to classification techniques when one wants to predict a class. 

**Implementation strategy and performance measures:** This involves minimising cost function using techniques such as gradient descent. 

## Linear

**Prediction equation**

$$
\begin{equation}\hat{y} = h_\theta(\bar{x}) = \theta^T\cdot\bar{x}
\end{equation}
$$

where

$\bar{x}$: instances feature vector length of n with $x_0$ = 1 the bias term 

$\theta^T$: transpose of model parameter vector with bias term $\theta_0$

$h_\theta$: hypothesis function

**Cost function (Mean Square Error)**

Use Root Mean Square Error (RMSE) for performance measure.

Also known as the *cost function*

$$
\begin{equation}
MSE({\bar{x}, h_\theta}) \equiv  MSE({\bar{x}}) = \frac{1}{m}\sum_i\left(\theta^T\cdot \bar{x}^{(i)} - y^{(i)}\right)^2
\end{equation}
$$

**Normal equation (closed form solution to predict model parameters)**

$$
\begin{equation}
\hat{\theta} = \left(\bar{X}^T\cdot \bar{X} \right)^{-1}\cdot \bar{X}^T\cdot \bar{y}
\end{equation}
$$

where $\hat{\theta}$ value of $\theta$ that minimises the cost function.

## Polynomial regression

Add the degree of the polynomial as another feature to fit. Use *PolynomialFeatures* class from scikit-learn to implement polynomial features properly. It’ll also capture all the cross terms of the type $x^k_i\times x^{m-k}_j$ for a polynomial of degree $m$. Be aware of the combinatorial explosion of the features due to cross terms.

## Regularised regression models

Deals with the problem of overfitting by regularising the model (i.e. constraining the set of features) e.g. degrees of polynomial fit or weights in linear regression. 

Most regularised models need scaled data. 

### Ridge regression (Tikhanov regularisation)

Aim: keep the weights as small as possible.

**Cost function**

$$
\begin{equation}
J(\theta) = MSE(\theta) + \alpha \sum_{i = 1}^n \theta^2_i
\end{equation}
$$

where $\alpha$ is another hyper-parameter.

**Normal equation**

$$

\begin{equation}
\hat{\theta} = \left(\bar{X}^T\cdot \bar{X}+ \alpha\bar{I} \right)^{-1}\cdot \bar{X}\cdot\bar{y}
\end{equation}
$$

where $\bar{I}$ is $n\times n$ identity matrix. 

### Lasso regression: Least operator shrinkage and selection operator regression

Tends to completely eliminate the weight of the least important features, i.e. performs feature selection and outputs a *sparse* model. 

**Cost function**

$$
\begin{equation}
J(\theta) = MSE(\theta)+ \alpha\sum_{i=0}^n|\theta_i|
\end{equation}
$$

$|\theta|:$  $l_1$ norm of the weight vector

### Elastic net regression

## Logistic Regression

## 

# Performance measures

## Minimise the RMSE

Technique involves variation of gradient descent. 

- Batch gradient descent: compute gradient on the entire training dataset
- Stochastic gradient descent: compute only on one particular instance of the training dataset
- Mini-batch gradient descent: compute on small batch (sets of random instances) of the training dataset

Check RMSE as a function of training set size.

Common problems are 

- speed/memory when computing gradient descent
- overfitting/underfitting model

# Implementation techniques

sklearn class: *sklearn.linear_model*

##