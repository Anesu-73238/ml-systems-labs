Problem Definition

Task: The goal of this project is to build a machine learning model that can predict whether a person earns more than $50,000 per year based on information about them, such as their age, education, job, and working hours. We will use the UCI Adult Census Income dataset, which is available through fetch_openml(name='adult', version=2).

Input data: The dataset contains 14 features describing each person. These include age, education, occupation, marital status, working hours per week, and other personal information. Some values are numerical, while others are categorical. There are also some missing values marked with ? that need to be handled during data preprocessing.

Output: The model will predict one of two income categories: whether a person earns more than $50,000 per year or earns $50,000 or less. Since fewer people in the dataset belong to the higher-income group, we cannot rely on accuracy alone to evaluate how well the model performs.

Success metrics: We will evaluate the model using two metrics: accuracy and F1-score. The F1-score is particularly important because it helps us understand how well the model identifies people earning above $50,000. The evaluation will be done using a separate 30% portion of the dataset that will not be used to train the model or prepare the data.

Constraints: To make sure the results are reliable, the dataset will be divided into training and testing sets before any preprocessing takes place. Missing values, outliers, and categorical data will be handled using information from the training set only, so the model does not accidentally learn from the test data. We will also use a fixed random seed to make the results reproducible. All settings will be stored in a config.yaml file, the processed dataset will have a SHA256 hash recorded, and the entire project should be reproducible with a single command. Data quality issues will be identified and reported rather than silently removing rows or changing values.

Limitations and risks: The dataset comes from the 1994 U.S. Census and includes sensitive information such as race, sex, and country of origin. Because of its age and the sensitive nature of these attributes, the model is intended for learning and testing a machine learning pipeline, not for making real-world decisions about people’s income, employment, or opportunities.