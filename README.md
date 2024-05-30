# Depression Prediction using ML Algorithms

## Purpose

This project aims to build a text classification pipeline that preprocesses, augments, and cleans textual data, and then trains machine learning models to classify the text.
Functionality

    Data Preprocessing: Cleaning and preparing text data by removing URLs, emojis, mentions, hashtags, numbers, and applying tokenization and lemmatization.
    Data Augmentation: Enhancing the dataset using text augmentation techniques.
    Feature Extraction: Transforming text data into numerical features using techniques like Count Vectorization.
    Model Training: Training machine learning models, including Decision Tree and Gradient Boosting classifiers, to classify the text.
    Model Evaluation: Evaluating model performance using metrics such as accuracy, precision, recall, and F1 score.
    Hyperparameter Tuning: Optimizing model performance through hyperparameter tuning using Bayesian optimization with TuneSearchCV.

## Preprocessing

    Cleaning: Removing URLs, emojis, mentions, hashtags, numbers, and non-alphanumeric characters.
    Normalization: Converting text to lowercase.
    Tokenization: Splitting text into individual tokens.
    Lemmatization: Reducing words to their base forms.
    Stopwords Removal: Eliminating common words that do not contribute to classification.

## Data Augmentation Using SMOTE

In the context of text classification, data augmentation refers to techniques used to increase the diversity and size of the dataset by generating new synthetic data points. One common challenge in text classification is dealing with class imbalance, where one class may have significantly fewer samples than others. Synthetic Minority Over-sampling Technique (SMOTE) is a popular method for addressing this issue.

## What is SMOTE?

SMOTE is an oversampling technique specifically designed to deal with imbalanced datasets. It works by creating synthetic samples of the minority class by interpolating new instances between existing minority class samples. This helps to balance the class distribution and improve the performance of machine learning models, particularly in scenarios where one class is underrepresented.

## Machine Learning Models

    Decision Tree Classifier: Trained with SMOTE to handle class imbalance and tuned using Bayesian optimization.
    Gradient Boosting Classifier: Evaluated for performance on the test dataset.

## Results

    Validation Precision: Achieved the highest precision during hyperparameter tuning.
    Test Metrics:
        Precision: High precision score indicating a low false positive rate.
        Recall: Measures the model's ability to capture positive instances.
        F1 Score: Balance between precision and recall.
        Accuracy: Overall correctness of the model.

## Conclusion

The text classification pipeline successfully preprocesses and augments textual data, allowing machine learning models to achieve high performance. The combination of SMOTE for balancing the dataset and Bayesian optimization for hyperparameter tuning results in a robust classification system.
