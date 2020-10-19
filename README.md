# CSCI297-Test2

Danesh Badlani, Sam Bluestone, and Leslie Le

## Design Decisions

We decided to categorize the 'Race' feature using pandas' get_dummies method which allows us to pass in a column of a DataFrame as a parameter and converts each value  in that column to its own separate column. For example, if the subject is White, the new 'White' column will measure to 1, and the rest of the new columns (latinx, African American, Asian) will be 0.

## EDA

For the data to help with classification, we decided to binarize the 'Chance of Admit ' feature by setting a threshold at 0.82. If the feature were above that threshold, it would b e categorized as 1, and if not, it will be 0. 

We decided to use the 'GRE', 'TOEFL', and 'CGPA' features because those three features were the most highly coorelated features compared to the others, with the coorelation hovering around 0.82 or 0.83.

### Problematic or Biased Features?

## Best Performing Model

## Other Models

We decided to not consider the Linear Regression Model because instead of predicting on target classes, it predicts continously. However, since we want to be able to classify the chances of admission, then Linear Regression would not be the best model to do so.

For the decision tree, we decided to use the three features in classification. We also had a depth of 3/4(?) because it allowed the tree to consider more features in its modeling. With the depth of two, the tree only considered the CGPA feature. With a depth of three or greater, the tree begins to consider more features - here specifically it considers the TOEFL feature. Although the increase in depth decreased the accuracy, it was necessary to avoid the tree from becoming overgeneralized on a single feature. However, even though the accuracy dropped, it still remains above 90%.

## Why can accuracy not be the most appropriate metric to make the sole decision by?

### Resources

* pandas API
* sklearn API