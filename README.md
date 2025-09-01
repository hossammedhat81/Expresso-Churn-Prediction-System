## Description:

Expresso is an African telecommunications company that provides customers with airtime and mobile data bundles. The objective of this challenge is to develop a machine learning model to predict the likelihood of each Expresso customer “churning,” i.e. becoming inactive and not making any transactions for 90 days.

This solution will help Expresso to better serve their customers by understanding which customers are at risk of leaving.

Expresso is an African telecommunications services company that provides telecommunication services in two African markets: Mauritania and Senegal. Expresso offers a wide range of products and services to meet the needs of customers.

The objective of this hackathon is to develop a predictive model that determines the likelihood for a customer to churn - to stop purchasing airtime and data from Expresso.

### Files:

- Train.csv - contains information about 2 million customers. There is a column called CHURN that indicates if a client churned or did not churn. This is the target. You must estimate the likelihood that these clients churned. You will use this file to train your model.
- Test.csv - is similar to train, but without the Churn column. You will use this file to test your model on.
- SampleSubmission.csv - is an example of what your submission should look like. The order of the rows does not matter but the name of the user_id must be correct.

## Evaluation

The evaluation metric for this challenge is Area Under the Curve (AUC).

The values can be between 0 and 1, inclusive. Where 1 indicates the customer churned and 0 indicates the customer stayed with Expresso.