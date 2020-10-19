# CSCI297-Test2

Danesh Badlani, Sam Bluestone, and Leslie Le

## Design Decisions

We decided to categorize the 'Race' feature using pandas' get_dummies method which allows us to pass in a column of a DataFrame as a parameter and converts each value  in that column to its own separate column. For example, if the subject is White, the new 'White' column will measure to 1, and the rest of the new columns (latinx, African American, Asian) will be 0.

## EDA

For the data to help with classification, we decided to binarize the 'Chance of Admit ' feature by setting a threshold at 0.82. If the feature were above that threshold, it would b e categorized as 1, and if not, it will be 0. 

We decided to use the 'GRE', 'TOEFL', and 'CGPA' features because those three features were the most highly coorelated features compared to the others, with the coorelation hovering around 0.82 or 0.83.

### Problematic or Biased Features?

Some problematic features were when we didn't have a certain feature for a subject. For example, in the given csv file, we have a missing feature in the Chance of Admit feature. We also have missing data from subjects about their race, so what our group did was we discarded the subjects that did not have all the data. 

The 'Race' feature was problematic as well because it was a categorical variable, so you should not carelessly assign values (such as 1, 2, etc.) to each category. If you do this, it will skew the coorelation values. 

## Best Performing Model

## Other Models

We decided to not consider the Linear Regression Model because instead of predicting on target classes, it predicts continously. However, since we want to be able to classify the chances of admission, then Linear Regression would not be the best model to do so.

For the decision tree, we decided to use the three features in classification. We also had a depth of 3 because it allowed the tree to consider more features in its modeling. With the depth of two, the tree only considered the CGPA feature. With a depth of three or greater, the tree begins to consider more features - here specifically it considers the TOEFL feature. Although the increase in depth decreased the accuracy, it was necessary to avoid the tree from becoming overgeneralized on a single feature. However, even though the accuracy dropped, it still remains above 90%.

## Why can accuracy not be the most appropriate metric to make the sole decision by?

Accuracy should not be the sole metric you sue to decide which model performs the best against others. Accuracy is very dependent on the data you are processing, so if you have a dataset that leans heavily towards one class during training and testing, the model will have a high accuracy as a result of the skewed data. To get the best  results for accuracy, you would want a balanced data set, however, that is not always possible. 

Accuracy is also an easy metric to alter. For example, if the threshold were to be  adjusted, one way or the other, predictions that were previously classified as a certain class may change to another. This change across the entire dataset will change the accuracy. One would be able to fib their comparisons between models if they only used accuracy and adjusted the threshold of a singular model in a favorable way.

In total, we used accuracy, f1 score, precision score, and the recall score to analyze each models' performance.

### Resources

* pandas API
* sklearn API