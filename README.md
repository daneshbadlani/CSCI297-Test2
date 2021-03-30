# Graduate Admission Prediction

Danesh Badlani, Sam Bluestone, and Leslie Le

## Design Decisions

For the data, to help with classification, we decided to binarize the 'Chance of Admit ' feature by setting a threshold at 0.82. If the feature were above that threshold, it would be categorized as 1, and if not, it will be 0. The threshold of 0.82 was chosen because it represents the thrid quartile of the feature.

We decided to categorize the 'Race' feature using pandas' get_dummies method which allows us to pass in a column of a DataFrame as a parameter and convert each value in that column to its own separate column. For example, if the subject is White, the new 'White' column will measure to 1, and the rest of the new columns (latinx, African American, Asian) will be 0.

## EDA

There was some missing data, and we decided to remove all the missing values using dropna method because we wanted to keep only original values instead of synthetically creating data at points where there is missing data. After removing all the missing values, we were left with 356 rows to deal with eradicating only about 10% of initial data. We found out that there was a positive corelation between 'GRE' and 'TOEFL' scores as well as 'GRE' and 'CGPA'. We decided to use the 'GRE', 'TOEFL', and 'CGPA' features for our models because those three features were the most highly coorelated features compared to the others with 'Chance of Admit', with the coorelation scores of 0.80, 0.79 and 0.87 respectively. We split the dataset into 80% train data and 20% test data using the train_test_split function because it eventually ended giving us better overall results for all the different models. We also used the Standard Scaler to standardize our data for some of the models because the distribution of the features is normal as shown in scatterplot.

### Problematic or Biased Features?

The 'Race' feature is problematic because it is a categorical variable, so you should not carelessly assign values (such as 1, 2, etc.) to each category. If you do this, it will skew the coorelation values by assigning incorrect values to the features based on the carelessly assigned vairables. The correlation matrix confirmed this, showing that none of the races had any correlation with 'Chance of Admit'.

However, the only model that did not throw out these features was Naive Bayes because the Naive Bayes model assumes that all features are independent. Thus, the correlation between features is not important in determining whether or not to include them in the final model.

## Results

#### Linear Regression

- Slope: 0.002
- Intercept: -1.606
- R^2 Score: 0.776
- MSE: 0.040

#### Logistic Regression

- Logistic Regression Accuracy: 0.958
- Logistic Regression F1-Score: 0.941
- Logistic Regression Precision: 0.960
- Logistic Regression Recall: 0.923

#### Decision Tree

- Decision Tree Accuracy: 0.931
- Decision Tree F1-Score: 0.898
- Decision Tree Precision: 0.957
- Decision Tree Recall: 0.846

#### Random Forest

- Random Forest Accuracy: 0.958
- Random Forest F1-Score: 0.939
- Random Forest Precision: 1.000
- Random Forest Recall: 0.885

#### SVM

- SVM Accuracy: 0.958
- SVM F1-Score: 0.941
- SVM Precision: 0.960
- SVM Recall: 0.923

#### KNN

- KNN Accuracy: 0.917
- KNN F1-Score: 0.870
- KNN Precision: 1.000
- KNN Recall: 0.769

#### Naive Bayes

##### Gaussian

- Naive Bayes accuracy: 0.943966
- Naive Bayes F1: 0.866346
- Naive Bayes precision: 0.830379
- Naive Bayes recall: 0.923333

##### Bernoulli

- Naive Bayes accuracy: 0.905049
- Naive Bayes F1: 0.801553
- Naive Bayes precision: 0.694921
- Naive Bayes recall: 0.970833

### Other Models

We decided to not consider the **linear regression model** because instead of predicting on target classes, it predicts continously. However, since we want to be able to classify the chances of admission, then Linear Regression would not be the best model to do so.

For the **decision tree**, we decided to use the three features in classification. We also had a depth of 3 because it allowed the tree to consider more features in its modeling. With the depth of two, the tree only considered the CGPA feature. With a depth of three or greater, the tree begins to consider more features - here specifically it considers the TOEFL feature. Although the increase in depth decreased the accuracy, it was necessary to avoid the tree from becoming overfitted based on a single feature. However, even though the accuracy dropped, it still remains above 90%. We ended up using "Gini" impurity as our criterion because we wanted to minimize the probability of misclassification and "entropy" criterion produced similar results.

In the **random forest**, we decided to use the 'entropy' criterion because it allowed us to maximize the metrics as compared to the 'gini' criterion. We used about 25 trees for the forest because adding too much seemed to lower the metric scores. Even though having a smaller or larger value for the n_estimators did not change the metric scores, we decided against that because that may potentially cause our model to be overfitted. The random state does not matter. The n_jobs value did not change the scores, so we chose 2.

The **SVM model** uses the default rbf kernel trick, which is a Gaussian kernel that provided us the best metrics overall to the level of logisitic regression. We also tried bunch of different C values, the regularization parameter, where 10.0 gave us the optimal results. The random state did not matter.

For the **KNN model**, we decided to have 5 neighbors. You can also see within the KNN file, there is a method that allows you to test plethora of values for the k value. There you can see that when k equals 5, all metrics are maximized. We also used the chebyshev distance metric for the model. With this metric, it gave us high scores in all areas compared to the other three metrics. Euclidean and chebyshev distance metrics performed the same though.

For the **Naive Bayes model**, the 5 options for models to use were: Gaussian, Bernoulli, Multinomial, Complement, and Categorical. Multinomial, Complement, and Categorical data can be ruled out immediately because they are meant for discrete datasets, which we clearly do not have. Bernoulli works similarly to Multinomial, but instead of assuming we have discrete features, it binarizes the data before fitting it. For that reason, we can try it out. Gaussian, on the other hand, works better with continuous data than the others. For the Gaussian model, the only two Looking at the results, the Gaussian model fared better in every metric we measured with the exception of recall. The Gaussian model most likely works better than Bernoulli because Bernoulli requires more transforming of the data. For additional hyperparameters for Gaussian NB, we have priors and var_smoothing. Priors are used if we have prior probabilities determined. Because we did not calculate any prior probabilities, we can just use the model to determine the probabilities. Changing the var_smooting parameter did not change the results, so it was left at the default value. For additional hyperparameters for the Bernoulli model, we have alpha, binarize, fit_prior, and class_prior. We don't chance fit_prior and class_prior for the same reason we didn't change the priors parameter. Changing alpha and binarize did nothing to the Bernoulli results, so we left them at their default values.

## Why can accuracy not be the most appropriate metric to make the sole decision by?

Accuracy should not be the sole metric you use to decide which model performs the best against others. Accuracy is very dependent on the data you are processing, so if you have a dataset that leans heavily towards one class during training and testing, the model will have a high accuracy as a result of the skewed data. To get the best results for accuracy, you would want a balanced data set, however, that is not always possible.

Accuracy is also an easy metric to alter. For example, if the threshold were to be adjusted, one way or the other, predictions that were previously classified as a certain class may change to another. This change across the entire dataset will change the accuracy. One would be able to fib their comparisons between models if they only used accuracy and adjusted the threshold of a singular model in a favorable way.

In total, we used accuracy, F1 score, precision score, and the recall score to analyze each models' performance.

### Conclusion

There was some debate between choosing between the logistic regression model, SVM, and the random forest, but in the end, logistic regression was deemed the best. Although this model did not have the highest metrics in all categories, it was consistent with its performance with all the metrics measured. For example, in the random forest, the random forest a high F1 score and an absurd precision score, but the forest's recall score was 0.885. Even though logistic regression does not have that level of precision, it had a higher F1 score because its recall was on the same level as its precision, hovering around 0.94.

There was a large difference between random forest's scores: |precision (1.00) - recall (0.885)| = 0.115. However, logistic regression was level with its scores: |precision (0.960) - recall (0.923)| = 0.037.

With the **logistic regression model**, we used three features to help with the classification of admission. We decided on a C value of 100 because that allowed us to maximize the metrics, and C at 100 was a balance between allowing points to be erroneous but not having too many. The random state of the model does not matter. The type of algorithm used to optimize the model did not matter much because out of the four algorithms the model could use (newton-cg, sag, saga, lbfgs), due to the multiple features being considered, they all concluded with the same metric scores, and we chose 'lbfgs' in the end. We also chose a multi-class option of 'ovr' because we want to classify a binary value (admitted or not) for each feature that was considered.

SVM and logistic regression did have the same performance across the four metrics that we measured, but we determined that logisitc regression is the better model to use over SVM because it is a less computationally expensive model. This decision could be re-evaluated if we gather more data and find that SVM yields better results. Additionally, with the way we are attempting to classify the data, the logisitic regression model would be a better fit because we are basically classifying between two classes. A SVM model would be more useful if we  did not binarize the  Chance of Admit feature.

### Resources

- M. S. Acharya, A. Armaan and A. S. Antony, "A Comparison of Regression Models for Prediction of Graduate Admissions," 2019 International Conference on Computational Intelligence in Data Science (ICCIDS), Chennai, India, 2019, pp. 1-5, doi: 10.1109/ICCIDS.2019.8862140.
- pandas API
- sklearn API
