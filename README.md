# Amazon Video Game Review Sentiment Classification & Topic Modeling

## Project Overview

This repository contains a comprehensive social media analysis pipeline for video game reviews, featuring both supervised classification and unsupervised topic modeling. The project includes data preprocessing, feature extraction, sentiment classification using multiple machine learning algorithms, and topic discovery using Structural Topic Modeling (STM).

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
│   ├── svm_model.rds                  # Trained SVM model
│   └── stm_model_K4.rds               # Trained STM topic model (K=4)
├── preprocessing_review_data.ipynb    # Data preprocessing notebook
├── review_classifier.R                # Model training and evaluation script
├── topic_modeling.R                   # Topic modeling analysis script
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

### 3. Topic Modeling (`topic_modeling.R`)

The R script performs unsupervised topic discovery using Structural Topic Modeling (STM):

#### Topic Discovery
- **Data Preparation**: Uses the same preprocessed review data from `game_reviews.csv`
- **Document Processing**: Converts text to document-term matrix format compatible with STM
- **STM Implementation**: Applies Structural Topic Modeling with K=4 topics
- **Topic Labeling**: Extracts top words for each topic using FREX scoring

#### Topic Visualization
- **Word Clouds**: Creates FREX-based word clouds for each of the 4 discovered topics
- **Topic Interpretation**: Provides interpretable topic labels and key terms
- **Model Persistence**: Saves trained STM model for future analysis

#### Key Features of Topic Modeling
- **Unsupervised Learning**: Discovers latent topics without predefined categories
- **FREX Scoring**: Uses FREX (Frequency + Exclusivity) for better topic word selection
- **Visualization**: Generates word clouds to visualize topic content
- **Reproducible**: Uses fixed random seed for consistent results

## Key Features

- **Balanced Dataset**: Equal representation of positive and negative reviews
- **Robust Text Processing**: Comprehensive cleaning and normalization
- **Multiple ML Algorithms**: Comparison of Naive Bayes, SVM, and Random Forest
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and feature importance
- **Topic Discovery**: Unsupervised topic modeling to identify latent themes
- **Reproducible Results**: Set random seeds for consistent outputs

## Dependencies

### Python (for preprocessing)
- pandas
- numpy
- json
- gzip
- re

### R (for modeling and topic analysis)
- tidyverse
- tidytext
- quanteda
- caret
- e1071
- randomForest
- ROCR
- pROC
- ggplot2
- stm
- wordcloud2
- ggwordcloud
- Matrix

## Results

The project generates several outputs:

1. **Processed Datasets**: Clean, balanced review data ready for analysis
2. **Trained Models**: Saved Random Forest, SVM, and STM models for future predictions
3. **Classification Evaluation Visualizations**:
   - ROC curve comparisons
   - Confusion matrix heatmaps
   - Feature importance plots
4. **Topic Modeling Visualizations**:
   - FREX-based word clouds for each discovered topic
   - Topic labels and key terms
   - Interpretable topic themes

## Model Performance

The project provides comprehensive evaluation metrics for:
- **Classification Models**: Direct comparison of Naive Bayes, SVM, and Random Forest performance
- **Topic Models**: Interpretable topic discovery with 4 distinct themes identified through STM

## License

This project is for educational purposes as part of a social media analysis course.

