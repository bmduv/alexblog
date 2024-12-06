---
title: ML Basics Review
draft: false
tags:
---
## Supervised Learning
Supervised learning consists of input values (independent variables) and target/output values (dependent variables) with the goal of creating function for mapping the input values to the output values accurately. It is called "supervised" learning because the model is provided with correct training, allowing it to make predictions/classifications on unseen data.

## Unsupervised Learning
Unsupervised learning consists of just input data with no output/target data. The goal of unsupervised learning is to find relationship/characteristics/patterns among the input data. Algorithms for unsupervised learning aim to cluster together similar data or find patterns that indicate the hidden characteristic of the data. Because there are no correct outputs provided for training, the algorithm explores the data with no prior knowledge.

## Reinforcement Learning
Reinforcement learning deals with an agent interacting with an environment to maximize its cumulative reward/minimize the penalty. The agent learns by taking actions in an environment and receiving feedback in the form of rewards or penalties. The goal is to learn the optimal policy - a set of actions that maximize the cumulative reward over time.

## Steps in ML Pipeline
Here are the following steps for creating a machine learning pipeline:
1. Data Collection
	- Pull/collect required data 
2. Data Preprocessing
	- Data balancing to prevent bias, data normalization, cleaning up missing values...
3. Feature Engineering 
	- Transforming raw data into useful features - this can consist of handpicking relevant features, new feature creation, categorical feature encoding...
4. Data Splitting 
	- Split the new data into train/valid/test data
5. Model Selection 
	- Choose the optimal model (regression, classification...) depending on specific use cases and available resources involving scalability, efficiency, and interpretability.
6. Model Training 
	- Train the model where it learns the patterns and relationships among input variables to make more accurate outputs. This learning is done through optimization algorithms that update the weight of the model based on the current loss in order to minimize the loss.
7. Model Evaluation 
	- Model is evaluated on performance and generalization ability. Common metrics used are, accuracy, recall, precision, F1 score, and mean squared error.  
8. Model Optimization 
	- If the model is not performing as well, further optimization can be done including hyperparameter tuning, new feature creation, different regularization methods, ensemble methods, etc.
9. Model Deployment
10. Model Maintenance

## Generative Models
Generative models aim to model the probability distribution of the input data. They learn the joint probability distribution between features and output labels (if available). The model tries to understand how the data is generated and capture the underlying patterns and dependencies among features. Once this has been learned, the model can generate new samples that resemble the original data. Some of these example models include Gaussian Mixture Model, Hidden Markov Model, and Variational Autoencoders.

We aim to model the joint probability distribution of input features ($X$) and output labels ($Y$). The goal is to learn the join probability distribution and learn the relationship/dependencies between $X$ and $Y$.
- Mathematically, the join probability can be written as: 
$$
P(X, Y) = P(Y) * P(X|Y)
$$
   where $P(Y)$ represents prior probability of $Y$ and $P(X|Y)$ represents the conditional probability of $X$ given $Y$. Generative models learn both $P(Y)$ and $P(X|Y)$.

## Discriminative Models
Discriminative models aims to learn the decision boundaries that separate different classes or categories. They learn the conditional probability of the output labels given the input features $P(Y|X)$. Such models include Support Vector Machine, Logistic Regression, and Deep Neural Network.

## Generative vs. Discriminative
- Objective: Generative Models aim to model the joint probability distribution of both input features and output labels. Discriminative Models aim to model the conditional probability distribution of the output labels given the input features.
- Data Generation vs. Decision Boundary: Generative Models aim to understand the data generation process and learn the patterns and dependencies between the input features and label outputs by learning the joint probability distribution. Because of this, Generative Models are capable of generating new samples that are similar to the original data. Discriminative Models aim to learn the decision boundary that separates difference labels/classes/categories by learning the probability distribution of the output labels.
- Training: Generative Model are generally more expensive to train as it needs to learn to estimate both the prior probability $P(Y)$ and conditional probability $P(X|Y)$. Discriminative Models directly estimate the conditional probability $P(Y|X)$.

## Bias vs. Variance
- Bias Error: A model with high bias tends to oversimplify the underlying patterns in the data and make strong assumptions about the relationships between the features and output variable. This results in underfitting of the training data as the model fails to capture the complex relationships/variations among the data, ultimately leading to high training error.
- Variance Error: High variance refers to the instability/high fluctuations of the model's predictions caused by small changes in the training data. A model with high variance is overly complex and sensitive to noise and randomness in the training data. As a result, the model can fit closely to the training data, but fails to generalize well to unseen data.

As bias is decreased (model is more complex and flexible), the model is often better at fitting the training data. However, this can lead to high variance where the model can become to complex and becomes too sensitive to any noise or changes to the training data
As variance is decreased (model is simplified), the model becomes less sensitive to the specific details of the training data and becomes better at generalizing to unseen data. However, this can lead to increase in bias where the model fails to capture the patterns/relationships in the train data.

## Regularization
Regularization helps prevent overfitting of the model and helps the model improve in generalizing. This is done by adding a regularization term to the loss function in which certain characteristics of the model will be penalized and encourage simplicity to reduce the impact of noisy/irrelevant features. By adding this regularization term along with the regularization parameter $a$ to control the strength of the regularization, the model can be trained to fit the data while keeping its complexity in check. 
- Overfitting Prevention: Overfitting occurs when the model fits the training data too closely, capturing noise and irrelevant patterns that are specific to the training set. Regularization helps prevent this by penalizing model complexity, discouraging it from memorizing noise.
- Generalization Improvement: By constraining the complexity of the model, regularization helps the model to focus on the underlying patterns and dependencies applicable to the entire dataset.
- Simplicity: Due to the penalizing nature of regularization, the model is encouraged to be simpler, and thus more interpretable.
- Bias-Variance Tradeoff: By having a regularization term , we can tune the model to be simpler and may lower its ability to precisely fit the training data (high bias), but can help reduce the sensitivity of the model to noise/irrelevant data and improve performance on unseen data (low variance).
- Parameter Shrinking: Regularization can drive some of the parameters towards zero. This can help reduce the impact of less important features, preventing them from having too much impact on the model's predictions. It effectively performs feature selection by assigning smaller parameters to less important features.
$$
J_{reg} = J + α * R(w) 
$$
### L1 Regularization (Lasso)
One of the most commonly used regularization term, L1 adds the sum of the absolute values of the model weights as a penalty term to the loss function. This penalty encourages sparsity in the parameter vector (can drive some parameters to zero): 
$$
R(w) = ||w||_1 =  |w_1| + |w_2| + ... + |w_n| 
$$
### L2 Regularization (Ridge)
L2 adds the sum of the squared values of the model's parameters as a penalty term to the loss function. It promotes smaller parameter values:
$$
R(w) = ||w||_2^2 =  w_1^2 + w_2^2 + ... + w_n^2
$$
### L1 vs. L2
- Effect on model behavior:
	- L1 encourages sparsity in model parameters by pushing some parameters to zero. This results in L1 acting as a feature selection in which some features are ignored, leading to a more compact and interpretable model.
	- L2 penalizes larger parameter values and encourages smaller parameter values. Unliked, L1, L2 does not push parameters to be zero, instead, it reduces the impact of less important features but still keeps them in the model prediction. L2 helps to evenly distribute the weight values, reducing the influence of individual features and leading to smoother, stable models.
## Elastic Net Regularization
Elastic Net Regularization is a combination of L1 and L2 regularization, it addresses limitations of each technique by taking a compromise between feature selection (L1) and parameter shrinkage (L2).
$$
J_{reg} = J + α * \{λ * L1(w) + (1 - λ) * L2(w)\}
$$
## Regression Models
The goal of regression is to predict a continuous, numerical value. This can be defined as finding a function f(x) that maps an input feature vector x to a continuous output value y, where y is a real number. Regression can be considered a discriminative model as it focuses on predicting numerical values based on input features.
- Mean Squared Error (MSE): Calculates the average squared differences between the predicted values and actual values. MSE is widely used as it emphasizes large errors due to the squaring nature. Also, it is nonnegative due to squaring and differentiable, allowing gradient-based optimization. However, MSE is sensitive to outliers as their squared losses can dominate overall loss.
$$
MSE = \frac{1}{n}\sum_{i=1} ^{n}(y_i - \hat{y}_i)^2
$$
- Root Mean Squared Error (RMSE): It is the squared root of the MSE and provides a measure of average magnitude of the errors. It is commonly used as the errors are in the same scale as target variables.
$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$
- Mean Absolute Error (MAE): Calculates the average absolute difference between predicted and actual values. MAE is less sensitive to outliers as it does not involve squaring. However, it is not differentiable at zero and can be a challenge for gradient-based optimizations.
$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
$$
- R-Squared (Coefficient of Determination): Measures the proportion of the variance in the target variable that can be explained by the model. R-Squared ranges from 0-1, where higher values indicate better fit.
$$
\begin{gathered}
R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y}_i)^2} \\\\
\text{Where } \hat{y} \text{ is predicted value and } \bar{y} \text{ is mean of predicted value}
\end{gathered}
$$
## Classification Models
The goal is to predict a discrete, categorical outcome or label. Classification can be defined as finding a function f(x) that maps feature vector x to a discrete output class y, where y belongs to a set of possible classes. The objective is to find the decision boundary/decision function that separates different classes in the feature space. 
- Accuracy: Measure the overall correctness of the model, it calculates a ratio of correct predictions (true positive + true negative) to the total number of predictions
$$
Accuracy = \frac{TruePositve+TrueNegative}{TruePositive+TrueNegative+FalsePositive+FalseNegative}
$$
- Precision: Measures proportion of correctly predicted positive instances out of all instances predicted as positive.
$$
Precision = \frac{TruePositive}{TruePositive+FalsePositive}
$$
- Recall: Measures proportion of correctly predicted positive instances out of all actual positive instances, focuses on capturing all positive instances.
$$
Recall = \frac{TruePositive}{TruePositive + FalseNegative}
$$
- F1 Score: It is a harmonic mean between precision and recall, it shows a single metric that balances both precision and recall.
$$
F1Score = 2 * \frac{Precision * Recall}{Precision + Recall}
$$
- Specificity: Measures proportion of correctly predicted negative instances out of all actual negative instances, focuses on capturing all negative instances.
$$
Specificity = \frac{TrueNegative}{TrueNegative+FalsePositive}
$$
- Area Under the ROC Curve (AUC-ROC): Assesses the model's ability to discriminate between positive and negative instances across different classification thresholds. It calculates the area under the Receiver Operating Characteristic (ROC) curve. The ROC curve plots True Positive Rate against False Positive Rate at various threshold values. The AUC-ROC ranges from 0-1 with higher values indicating better classification performance. 

## SVM
Aims to find the hyperplane that separate two classes with the largest possible margin, the margin is the distance between the hyperplane and the nearest data points from each class. 
$$
\begin{gathered}
w^Tx + b = 0 \\\\
\text{Where w is the normal vector to the hyperplane, } \\ \text{x is the feature  vector and b is the bias term}
\end{gathered}
$$
SVM selects the hyperplane that maximizes the margin while still correctly classifying training examples. To find the optimal hyperplane, SVM aims to minimize the norm of the weight vector ($||w||$) subject to the constraint that training examples are correctly classified: 
$$
\begin{gathered} 
minimize: \space (\frac{1}{2}) * ||w||^2 \\
subject \space to: \space y_i * (w^Tx_i + b) \ge 1
\end{gathered}
$$
The training examples that lie on or violate the margin are the support vectors, these support vectors play a crucial role in defining the decision boundary (used for finding distance between margin and data)
### Kernel Trick in SVM
This implicitly transforms features into higher-dimensional feature space without explicitly calculating the transformed features, it allows SVMs to handle nonlinearity and complex data distributions. Kernel tricks allows us to implicitly map input features to higher-dimensional feature space where the classes can become linearly separatable. This transformation is done by using a kernel function that calculates the similarity or inner product between two data points in higher-dimensional feature space. 
- Linear Kernel: $K(x, y) = x^Ty$
- Polynomial Kernel: $K(x, y) + (ax^Ty + c)^d$
- Gaussian (RBF) Kernel: $K(x, y) = exp(-\gamma||x-y||^2)$

## Logistic Regression
A popular algorithm used for binary classification, it models the relationships between the input features and the target variable belonging to a particular class. 
- Data Representation: Training data consists of input feature vector X and corresponding binary class labels y (0, 1).
- Logistic Function (Sigmoid): Models the probability of the class being in class 1.
  - The hypothesis of Logistic Regression is given by:
$$
\begin{gathered}
h\theta(x) = \sigma(\theta^tx) \\\\
\text{Where } h\theta(x) \text{ represents the predicted probability of the  target variable } \\ \text{in class 1 for the input feature vector x, with } \theta \text{ being the weights}
\end{gathered}
$$
  - Cost Function:
$$
\begin{gathered}
J(\theta) = -(\frac{1}{m}) * \sum_{i=1}^{m}[y_i * log(h\theta(x_i)) + (1-y_i) * log(1-h\theta(x_i))]
\end{gathered}
$$
