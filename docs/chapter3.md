# Chapter 3
In linear regression we assume that true  $f(x)$ is linear, or can be modeled by $f(x) = B_0 + B_1X_1 + ... B_pX_p$. We estimate $f(x)$ with $\hat f(x) = \hat B_0 + \hat B_1X_1 + ... \hat B_pX_p$.

---

## Estimating Coefficients
Simple linear regression is a parametric problem, in that we need to estimate parameters $B_0$ and $B_1$. 

Simple linear regression makes a line that is as close to datapoints as possible. We quantify closeness with RSS, defined as: 

$$RSS = \sum_{i = 1 }^{n}(Y_i - \hat f(x_i))^2 = \sum_{i = 1 }^{n}(Y_i - \hat B_0 - \hat B_1X_1)^2$$

We want to find the parameters that minimize RSS! The estimated parameters are

$$\hat B_1 = \frac{\sum_{i=1}^{n}(x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^{n}(x_i - \bar x)^2}$$

$$\hat B_0 = \bar y - \hat B_1 \bar x$$

where $\bar y$ and $\bar x$ are the averages over all x and y values respectively.

---

## Assessing Accuracy of Coefficient Estimates

> Hypothesis testing: Let there be a null hypothesis $H_0$ and an alternative hypothesis $H_a$. After running some tests, if we find that there are results that are statisticially significant in favor of $H_a$, then we reject $H_0$.

In the case of assessing coefficient estimates, we let $H_0: B_1 = 0$ and $H_a: B_1 \neq 0$. 

In order to judge whether or not $B_0$ might be 0, we can use a t-statistic, a measurement drawn from analysis of our estimations.

$$t = \frac{\hat B_1 - 0}{SE(\hat B_1)}$$

The t-statistic is a measurement of the number of standard deviations $\hat B_1$ is away from 0 using a student's t-distribution. If $t$ is large, it indicates that $B_1$ is likely non-zero. More precisely, if the _p-value_ associated with the t-statistic is large, it is likely that $B_1$ is non-zero. Generally we use $t > 2$ as our threshold for rejecting $H_0$.

---

## Assessing Model Accuracy

We've rejected $B_1 = 0$. Now we need to see how close our estimate is to the true $B_1$.

> Residual Standard Error: $RSE = \sqrt{\frac{1}{n-2}RSS}$; average amount that response will deviate from true regression line; is an estimate for standard deviation of $\epsilon$.

> $R^2$ statistic: $R^2 = \frac{TSS - RSS}{TSS}$; proportion of variance explained by a linear model w.r.t to $TSS = \sum(y_i - \bar y)^2$
- TSS is the total variability in the response with $\bar y$ as our predictor.
- RSS explains some of the variability.
- TSS - RSS is the variability that's left over after regression

$R^2$ has an interperetational advantage over RSE since its always between 0 and 1.

Can use confidence intervals:
- 95% confident that the true value of $B_1$ lies between [$\hat B_1 - 2SE(\hat B_1), \hat B_1 + 2SE(\hat B_1)$] since 2.5% of data lies above and below t = 2 in a student's t-distribution. 

---

## Multiple Linear Regression

Now, predictor is of the form $\hat Y = \hat B_0 + \hat B_1X_1 + ... \hat B_pX_p$

> $B_j$: average effect on Y for a unit increase in $X_j$ holding all other regressors fixed

_Why not just run a simple linear regression for each regressor?_

1. Can't see effect of 1 regressor in presence of others
1. Each equation will ignore other regressors
1. Can't account for correlations between regressors

We estimate regression coefficients by minimizing RSS, given by

$$RSS = \sum_{i = 1 }^{n}(Y_i - \hat f(x_i))^2 = \sum_{i = 1 }^{n}(Y_i - \hat B_0 - \hat B_1X_{i1} - ... - \hat B_pX_{ip})^2$$

### Correlation between regressors is an issue. Example from ISLR:

![](img/radioandnewspaper.png)

Newspaper advertising showed high correlation with sales in a simple linear regression setting. However, newspaper advertising's effect was not statistically significant in the presence of radio and TV advertising. In fact, since newspaper advertising was heavily correllated with radio advertising, it recieved the credit for radio advertising's effect (in the simple linear regression setting).

We can use t-statistics to find out whether or not a regressor has a statistically significant effect in the presence of other regressors. As seen in the above table from ISLR.

## Answering: Is there any relationship between any response and any predictor?

Let $H_0: B_1 = B_2 = ... = B_p = 0$ and $H_a: B_j \neq 0$ for some $j$.

We can compute the _F-statistic_. Then, we compute the corresponding p-value and then can discern whether or not we should reject the null hypothesis. **Note** we can't just use the F-statistic to reject $H_0$. 

$$F = \frac{(TSS - RSS) / p}{RSS/(n-p-1)}$$

## Answering: Is there any relationship between a subset of regressors and the predictor?

- Select a subset of size $q$ from the coefficients. 
- Run a hypothesis test where $H_0: B_{p-q+1} = B_{p-q+1} = ... = B_p = 0$.
- Compute an RSS_0 from a linear regression with all regressors except the $q$ removed. 
- Use $F = \frac{(RSS_0 - RSS) / q}{RSS/(n-p-1)}$ and use the associated p-value to determine the statistical significance of the subset of regressors.

**Note:** Still need to look at overall F-statistic even if we have individual p-values with for each regressor. When $p$ is large, we will have some (5%) $p$ values that are small _just by chance_. However, F-statistic adjusts for number of predictors.

---
## Deciding on Important Regressors

- **Forward Selection:**
  1. Begin with a model containing only the intercept
  1. Fit p simple linear regressions
  1. Add regressor with smallest RSS to model
  1. Repeat 2 - 4 (without repeating regressors) until some stopping rule satisfied
- **Backward Selection:**
  1. Begin with all regressors
  1. Run regression
  1. Remove variable with largest p-value
  1. Repeat 2 - 3 until stopping rule satisfied (i.e: all regressors meet a p-value threshold)
- **Mixed Selection:**
  1. Start with no regressors and only intercept
  1. Add regressors with minimum RSS
  1. Since p-values can increase as new regressors added to model, remove p-values that exceed a certain threshold
  1. Continue process until all regressors have sufficiently low p-value

---

## Model Fit

**Note:** $R^2$ will always increase when more regressors are added to the linear model. This is because it allows the model to more closely fit training data. One possible way to use this fact is if a regressors added to a model increases the model's $R^2$ by a miniscule amount, then it is possible that the variable does not have a statistically significant effect on the response when considering other regressors included in the model.

**Note:** $RSE = \sqrt{\frac{1}{n-p-1}RSS}$. Models with more regressors can have higher RSE if the decrease in RSS is small relative to the increase in $p$.

--- 
## Sources of Uncertainty in Prediction
1. Predicted least squares plane is only an estimate for true population regression plane. 
	1. We can use confidence intervals to determine how close $\hat Y$ is to $f(x)$
1. Model bias - Assuming data is linear when it might not be
1. Response can't be predicted perfectly because of irreducible error

### Prediction intervals
We can construct confidence intervals to quantify uncertainty surrounding a measurement over a large number of observations. We can use a prediction interval to quantify uncertainty for a specific prediction.

--- 

## **Other Considerations in Regression Model**
## **1. Qualitative Predictors**
Some regressors will be qualitative. We can capture the effect of these regressors using indicator variables. For example, if we wish to measure the effect of gender (a variable with only two levels _male_ and _female_), we can create a variable $x_i$, where 

$$x_i = \begin{cases} 

1 & male\\
0 & female

\end{cases}$$

Then, the regression becomes

$$y_i = B_0 + B_1x_i + \epsilon_i = \begin{cases} 

B_0 + B_1 + \epsilon_i & male\\
B_0 + \epsilon_i & female

\end{cases}$$

Which level gets assigned 0 or 1 is somewhat arbitrary. This dummy variable models the effect of the difference between male and female. For example, if the p-value of $\hat B_1$ is not statistically significant, then we can accept the null hypothesis and say that an observation being male or female doesn't affect output (the difference between male and female is not statistically significant). In this case, $B_0$ can be interpereted as the average output for females and $B_1$ as the average amount males are _above_ females.

However, choosing the numbers 0 and 1 has an effect on the interperetation of results. For example, we could have modeled the same data as

$$y_i = \begin{cases} 

B_0 + B_1 + \epsilon_i & male\\
B_0 - B_1 + \epsilon_i & female

\end{cases}$$

Now, $B_0$ is the overall average measure of output. $B_1$ tells us the amount that females are below the average and males are above the average. 

**Note:** Different choices for encodings lead to different interperetations.

We can also encode qualitative predictors with more than two levels by using multiple dummy variables. For example, if we want to encode an observation being either an Engineering, College, or Wharton student, we can let: 

$$x_{i1} = \begin{cases} 1 & \text{Engineering} \\0 & \text{not Engineering}\end{cases}$$

$$x_{i2} = \begin{cases} 1 & \text{College} \\0 & \text{not College}\end{cases}$$

Now, our regression model becomes

$$y = \begin{cases} 
B_0 + B_1 + \epsilon_i && \text{if Engineering} \\
B_0 + B_2 + \epsilon_i && \text{if College} \\
B_0 + \epsilon_i && \text{if Wharton}
\end{cases}$$

In this case, the Wharton category is the _baseline_. $B_0$ is the average output for Wharton students. $B_1$ is the average that Engineering students are above Wharton students. Finally, $B_2$ is the average amount (in units of response) that College students are above Wharton students.

## **2.Extending Linear Model**

We can model interaction effects between different regressors by adding an _interaction_ term. Let's say we begin with the model $Y = B_0 + B_1X_1 + B_2X_2 + \epsilon$. If $X_1$ and $X_2$ seem to have a synergistic effect, then we can add a third term $X_1X_2$ to the model. Our model now becomes $Y = B_0 + B_1X_1 + B_2X_2 + B_3X_1X_2 + \epsilon$. Doing so relaxes the additive assumption in the following way:

Note that we can re-write our model as $Y = B_0 + (B_1 + B_3X_2)X_1 + B_2X_2 + \epsilon$. Thus, a unit increase in $X_2$ actually affects the amount by which $X_1$ affects $Y$.

If the effect of the interaction term is statistically significant and the true model is not purely additive, we should see an increase in the $R^2$ value of the model with the interaction term included.

**Remember** the _hierarchical principle:_ include the base terms $X_1$ and $X_2$ in the model when we wish to include the interaction term $X_1X_2$ _even if_ neither $X_1$ nor $X_2$ have statistically significant p-values.

**Note:** to include interaction effects between qualitative and quantitative variables, we can simply multiply their regressors to form an interaction term as before.

#### Non Linear Relationships

Simply include some transformation of an existing regressor.

![](img/horsepowergraph.png)
![](img/horsepowertable.png)

## **3. Potential Problems**

1. Non-linearity of response-predictor relationships
1. Correlation of error terms
	1. Example of correlation in error terms time-series data
	1. Correlation is an issue: if there is correlation, estimated standard errors tend to underestimate true standard errors, resulting in confidence intervals that are narrower than they should be. 
1. Non-constant variance of error terms
	1. Can perform transformations on $\hat Y$ to make variance in error terms constant (when residuals increase w.r.t $x$). 
1. Outliers
	1. Possible that outliers may not have a large effect on least squares slope or intercept. However, they might have a significant effect on the RSE or $R^2$ statistic.
	1. Can identify outliers by looking at residual plots. Use _studentized residuals_ and if an observation has a residual $\geq 3$, we can classify as outlier (just a general rule).
1. High-leverage points
	1. Outliers are points for which $y_i$ is unusual given an $x_i$. High leverage points for which $x_i$ is unusual. Removing high-leverage points can have substantial impact on slope and intercept of least squares line. 
	1. Fairly easy to identify high-leverage points in simple linear regression.
	1. In multiple linear regression, it's important to look at predictors whollistically to find high leverage points.

		<img src="img/highleveragepoints.png" width="200" height="200" />

		Notice how the red point does not have an unusual value for neither $X_1$ not $X_2$. However, the combined $X_1,X_2$ value is unusual.
	1. Leverage statistic given by

		$$h_i = \frac{1}{n} + \frac{(x_i - \bar x)^2}{\sum_{j=1}^{n}(x_j - \bar x)^2}$$

		Noting that the average leverage for all observations is $\frac{p+1}{n}$, an observation with a leverage statistic that greatly exceeds $\frac{p+1}{n}$ could be classified as a high-leverage point.
1. Collinearity

---

## Comparing KNN with Linear Regression

>KNN Regression: Given a value for $K$ and a prediction point $x_0$, $KNN$ regression identifies the $K$ training observations that are closest to $x_0$ (represented by $N_0$). It then estimates $f(x_0)$ using the average of all training responses in $N_0$. Formally, $f(x_0) = \frac{1}{K}\sum_{x_i\in N_0}y_i$

Optimal value for $K$ will depend on the bias variance tradeoff. Usually, for small $K$ (i.e $K=1$), the variance is high but bias extremely low. The variance is high because any prediction (when $K=1$) will be based entirely on just one training observation. 

Parametric methods outperform non-parametric methods when parametric methods make correct assumptions of the form of the data. For example, if we fit a linear model to a fairly linear data-set, then a least squares regression might fit the data better than KNN regression. However, with increasing non-linearity, non-parametric methods, such as $KNN$ regression better fit the data. 

**Note:** $KNN$ regression suffers from the _curse of dimensionality_. As dimensions increase, the $N_0$ observations closest to $x_0$ become further and further away from $x_0$ as the number of dimensions increases. 

---