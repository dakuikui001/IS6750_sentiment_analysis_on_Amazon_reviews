# Amazon Video Game Review Classification

This project focuses on sentiment analysis of video game reviews using machine learning techniques. The goal is to classify reviews as either positive or negative based on their text content.

## Project Overview

This repository contains a complete pipeline for text classification of video game reviews, including data preprocessing, feature extraction, and model training with multiple machine learning algorithms.

## Dataset

The project uses the **Video Games 5** dataset from Amazon, which contains video game reviews with ratings and review text. 
Please download the original dataset from https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz and stored as `Video_Games_5.json.gz` in the `data/` directory.

## Project Structure

```
group_project/
├── data/
│   ├── Video_Games_5.json.gz          # Original Amazon video game reviews dataset
│   ├── game_reviews.csv               # Processed review data (6000 samples)
│   └── df_review.tf_label.csv         # Final dataset with TF-IDF features
├── model/
│   ├── random_forest_model.rds        # Trained Random Forest model
│   └── svm_model.rds                  # Trained SVM model
├── preprocessing_review_data.ipynb    # Data preprocessing notebook
├── review_classifier.R                # Model training and evaluation script
└── README.md                          # This file
```

## Workflow

### 1. Data Preparation (`preprocessing_review_data.ipynb`)

The Jupyter notebook handles the initial data preprocessing:

- **Data Loading**: Parses the compressed JSON file containing Amazon video game reviews
- **Text Cleaning**: 
  - Combines review title and text
  - Removes special characters and normalizes text
  - Filters reviews with length > 100 characters
- **Label Creation**: Converts 5-star ratings to binary labels:
  - Ratings 1-3: Negative
  - Ratings 4-5: Positive
- **Data Sampling**: Creates balanced dataset with 3,000 positive and 3,000 negative reviews
- **Output**: Generates `game_reviews.csv` with cleaned review data

### 2. Vector Creation and Model Training (`review_classifier.R`)

The R script performs feature extraction and model development:

#### Feature Engineering
- **Text Tokenization**: Converts reviews to individual words using `tidytext`
- **Stopword Removal**: Removes common English stopwords
- **TF-IDF Calculation**: Computes Term Frequency-Inverse Document Frequency weights
- **Document Filtering**: Removes words with document frequency < 20
- **Normalization**: Normalizes TF-IDF vectors by document length
- **Document-Term Matrix**: Creates sparse matrix representation for ML models

#### Model Training
The script trains three different classification models:

1. **Naive Bayes** (`e1071` package)
2. **Support Vector Machine (SVM)** (`e1071` package)
3. **Random Forest** (`randomForest` package)

#### Model Evaluation
- **Train-Test Split**: 70% training, 30% testing
- **Performance Metrics**: Confusion matrices, accuracy, precision, recall
- **ROC Analysis**: ROC curves and AUC comparison across models
- **Feature Importance**: Analysis of most important features (Random Forest)

## Key Features

- **Balanced Dataset**: Equal representation of positive and negative reviews
- **Robust Text Processing**: Comprehensive cleaning and normalization
- **Multiple ML Algorithms**: Comparison of Naive Bayes, SVM, and Random Forest
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and feature importance
- **Reproducible Results**: Set random seeds for consistent outputs

## Dependencies

### Python (for preprocessing)
- pandas
- numpy
- json
- gzip
- re

### R (for modeling)
- tidyverse
- tidytext
- quanteda
- caret
- e1071
- randomForest
- ROCR
- pROC
- ggplot2

## License

This project is for educational purposes as part of a social media analysis course.
