# Chapter 4: Classification
## Overview
We generally try to find the probabilities that a given observation $x_0$ belongs to a class $k$. 

---
## Pitfalls of Linear Regression
1. We can use linear regression to encode variables that have binary encodings (or just 2 levels). We can use linear regression to encode variables that have some inherent ordering or equal relative distance. Beyond these restrictions, however, it's difficult to use linear regression.
1. Linear regression produces a line, which can produce negative output or outputs greater than 1 for some $x$ values. This violates probabilities being on [0,1].

---
## Logistic Regression
Linear regression models predicted output given regressors. Logistic Regression models probability that the output is within a class given regressors.

### Logistic model
Let the odds of an event be defined as $\frac{p}{1-p}$. 

Let $logit(p) = ln(\frac{p}{1-p})$. We want to be able to map $logit(p)$ to some linear combinations of regressors. Let $ln(\frac{p}{1-p} = \alpha$, where $\alpha$ is the linear combination of regressors.

Note that $logit^{-1}(p) = \frac{e^{\alpha}}{1+e^{\alpha}}$. Notice that this function now has a range restricted to [0,1]. Let 

$$p(x) = \frac{e^{\alpha}}{1+e^{\alpha}} = \frac{e^{B_0 + B_1X_1}}{1+e^{B_0 + B_1X_1}}$$


Notice that $ln(\frac{p}{1-p}) = B_0 + B_1X$, a unit increase in $x$ yields a $B_1$ increase in the log-odds.

We estimate logistic coefficients by seeking estimates for $B_0$ and $B_1$ such that $P(x_i)$ yields close to 1 for observations of class 1 and close to 0 for observations of class 0. We find these coefficients by maximizing the likelihood function:

$$l(B_0,B_1) = \prod_{i:y_i=1}p(x_i) \prod_{j:y_j=0}(1-p(x_j))$$

We can measure accuracy of coefficient estimates by computing their standard errors, finding their z-statistics, and then finding their associated p-values. Large values for z-statistics indicates evidence against $H_0: B_1 = 0$.

---

## Multiple Logistic Regression

Now $ln(\frac{p}{1-p} = B_0 + B_1X_1 + ... + B_pX_p)$ and thus

$$p(x) = \frac{e^{\alpha}}{1+e^{\alpha}} = \frac{e^{B_0 + B_1X_1 + ... + B_pX_p}}{1+e^{B_0 + B_1X_1 + ... + B_pX_p}}$$

Keep in mind that we are still restricting each $X_i$ to be indicator variables. 

It's possible that we now find confounding between different variables in the multiple logistic regression setting. 

---

## Linear Discriminant Analysis
Important assumptions:
- Assume that all observations from a class $k$ follow a normal distribution.
- Shared variance of distribution of observations within each of $K$ classes. 

Note that the density function in one dimension is 

$$f_k(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{\frac{-1}{2\sigma^2}(x-\mu_k)^2}$$

We let $\pi_k$ be the proportion of training data points that lie within class $k$. In other words, $\pi_k$ is the probability that a randomly selected observation lies within the class $k$.

Remember that Baye's theorem states that $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$. From this, we see that 

$$P(Y=k|X=x) = \frac{P(X=x|Y=k)P(Y=k)}{\sum_{i=1}^{K}P(X=x|Y=i)P(Y=i)} = \frac{\pi_k f_k(x)}{\sum_{i=1}^{K}\pi_i f_i(x)}$$

Note that if $\pi_1f_1(x) > \pi_2f_2(x)$ then $log(\pi_1f_1(x)) > log(\pi_2f_2(x))$

Thus we can use the following linear equation to classify observations based on whether or not $\delta_1(x) > \delta_2(x)$ for some observation $x$ and two discriminant functions $\delta_1$ and $\delta_2$.

$$\delta_k(x) = log(\pi_k f_k(x)) = x\frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2\sigma^2} + log(\pi_k)$$

However, in order to find estimates for the discriminant functions, we need to find $\hat \pi_k$, $\hat \mu_k$, $\hat \sigma$. We can use 
- $\hat \pi_k = \frac{n_k}{n}$ where $n$ is the number of observations and $n_k$ is the number of observations within a class $k$.
-  $\hat \mu_k = \frac{1}{n_k}\sum_{i:y_i = k}x_i$.
- $\hat \sigma^2 = \frac{1}{n-K}\sum_{k=1}^{K}\sum_{i:y_i=k}(x_i - \hat \mu_k)^2$

We let the decision boundary be the $x$ value for which $\delta_1(x) = \delta_2(x)$. Thus we will classify $x$ in class 1 for values of $x$ for which $\delta_1(x) > \delta_2(x)$. In general, we would like the LDA decision boundary be as close to the Bayes Decision boundary as possible, since the Bayes Decision boundary is the theoretically optimal decision boundary.