# Chapter 2 Notes

## Fundamental Problem

$Y = f(x) + \epsilon$ is a function we don't know but want to estimate. $x$ is a regressor vector, $Y$ is our output, and $\epsilon$ is our _irreducible error_ that results for either missing regressors in $x$ or from unmeasurable variance in nature.

---

## Two classes of problems

> Prediction: Estimate $Y$ with $\hat{Y} = \hat f(x) + \epsilon$.

> Inference: see how $Y$ changes with changes in $X_1 ... X_p$

---

## Classes of Problems

_Regression_ problems deal with quantitative responses. 

1. parametric: reduces problem of estimating $f$ to a problem of estimating parameters
	1. i.e: When we want to use linear model $f(x) = B_0 + B_1X_1 + ... + B_pX_p$, we only have the estimate $B_0 + B_1 + ... + B_p$
	1. Generally more flexible - gets us further from true form of $f$
	1. More restrictive but generally more interperable than non-parametric models
1. non-parametric: 
	1. no assumption about shape/form of $f$
	1. tries to use $\hat f$ that is as close to data-points as possible
	1. Needs very large number of observations

_Classification_ problems deal with categorical or qualitative responses. Note that the type of predictor indicates what type of problem we are trying to solve.

---

## Assessing Model Accuracy
Measuring Quality of Fit - to quantify how "off" predictions are from true response data

We can use _mean squared error_

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{f}(x_i))$$ where $x_i$ is the ith observation.

- MSE is defined for both training and test data sets. 


![](img/trainingandtestMSE.png)

- **Note:** Can't just minimize training MSE to minimize test MSE. Most learning models work to minimize training MSE, which does not guarantee a smallest test MSE (might overfit to data). Flexible methods tend to have a higher chance of "overfitting"  training data, resulting in a higher test MSE.

---

## Bias-Variance Tradeoff

$$E[testMSE] = Var(\hat f(x)) + [Bias(\hat{f}(x_0))]^2 + Var(\epsilon)$$

> Bias: Error introduced by approximating real-life problem with simple models (i.e: linear model).

> Variance: the amount by which $\hat{f}$ would change if it was estimated using a different training data set; ideally, $\hat f$ shouldn't vary much across training sets 

- As model flexibility increases, bias generally decreases and variance increases
- As we increase flexibility, the bias initially decreases faster than the variance increases. At some point, the variance increases faster than the bias declines and thus the test MSE begins to increase.
- Note that test MSE will always be above $\epsilon$
- test MSE is minimized when sum of variance and bias is lowest
![](img/biasvariancetradeoff.png)

---

## Classification

### Training Error Rate
Use _training error rate_ to quantify accuracy of $\hat f$:

$$\frac{1}{n}\sum_{i=1}^{n}I(Y_i \neq \hat Y_i)$$
where $I$ is an indicator random variable that is 1 if $Y_i \neq \hat Y_i$ and 0 otherwise

Basically, training error rate averages all misclassifications across $n$ observations


### Test Error Rate

$Ave(I(Y_i \neq \hat Y_i))$ - a good classifier minimizes this

### The Bayes classifier
Assign a test observation $x_0$ a class $j$ for which $P(Y = j | X = x_0)$ is the highest.

> Bayes Error Rate: $1 - E(max_jPr(Y = j | X))$ - expectation just averages probability over all possible X
- Bayes Error Rate is analagous to irreducible error

### KNN

Brief Description: For an observation $i$, we look at the class of its $K$ nearest neighbors' classes. The class $k$ with the highest propotion wins and we assign $i$ the class $k$.

Classifier that classifies according to

$$Pr(Y=j|X=x) = \frac{1}{K}\sum_{i\in N_0}I(y_i = j)$$

- Note that low K means high model flexibility and high K means low model flexibility

![](img/lowKandhighK.png)

- We also observe the characteristic U shape as we increase model flexibility in the error rates in KNN

![](img/KNNerrorrates.png)

