# Social Media Analysis: Video Game Review Sentiment Classification & Topic Modeling

## Project Overview

This repository contains a comprehensive social media analysis pipeline for video game reviews, featuring both supervised classification and unsupervised topic modeling. The project includes data preprocessing, feature extraction, sentiment classification using multiple machine learning algorithms, and topic discovery using Latent Dirichlet Allocation (LDA).

## Dataset

The project uses the **Video Games 5** dataset from Amazon, which contains video game reviews with ratings and review text. 
Please download the original dataset from https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz and stored as `Video_Games_5.json.gz` in the `data/` directory.

## Project Structure

```
group_project/
├── data/
│   ├── Video_Games_5.json.gz          # Original Amazon video game reviews dataset
│   ├── game_reviews.csv               # Processed review data (6000 samples)
│   ├── game_reviews_for_topicModeling.csv  # Topic modeling dataset (20000 samples)
│   └── df_review.tf_label.csv         # Final dataset with TF-IDF features
├── model/
│   ├── random_forest_model.rds        # Trained Random Forest model
│   ├── svm_model.rds                  # Trained SVM model
│   └── sdm_model.rds                  # Trained LDA topic model (K=15)
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

The R script performs unsupervised topic discovery using Latent Dirichlet Allocation (LDA) with the `textmineR` package:

#### Data Preparation
- **Data Loading**: Reads preprocessed review data from `game_reviews_for_topicModeling.csv`
- **Text Lemmatization**: Uses `textstem::lemmatize_strings` to normalize words
- **Train-Test Split**: Partitions dataset into 50% training and 50% test sets
- **Document-Term Matrix**: Creates DTM using `textmineR::CreateDtm` with:
  - N-grams (unigrams and bigrams)
  - English and SMART stopword removal
  - Punctuation and number removal
  - Lowercase conversion

#### Topic Model Development
- **LDA Modeling**: Fits LDA model using `textmineR::FitLdaModel` (K=15 topics)
- **Model Evaluation Metrics**:
  - R-squared: Overall goodness of fit
  - Log Likelihood: Model convergence tracking
  - Probabilistic Coherence: Topic quality measurement
  - Topic Prevalence: Distribution of topics across corpus
- **Alpha Parameter**: Optimizes topic-topic distribution for balanced topics
- **Beta Parameter**: Controls word-topic distribution (0.01)

#### Topic Analysis
- **Top Terms**: Extracts top 20 terms for each topic based on word-topic probabilities
- **Topic Labeling**: Uses naive labeling algorithm based on probable bigrams
- **Coherence Visualization**: Creates histogram of coherence scores
- **Prevalence Visualization**: Scatter plot of topic prevalence vs. alpha parameters

#### Topic Visualization
- **Word Clouds**: Generates exclusivity-based word clouds for individual topics
  - Calculates exclusivity scores (topic-specific words vs. other topics)
  - Uses `ggwordcloud` for publication-ready visualizations
- **Topic Interpretation**: Provides interpretable topic labels and key terms

#### Test Set Application
- **Topic Prediction**: Applies trained LDA model to test set using Gibbs sampling
- **Topic Distribution Visualization**: Creates bar plots showing probabilistic topic distributions for specific reviews
- **Iterative Prediction**: Uses 1000 iterations with 50 burn-in iterations

#### Key Features of Topic Modeling
- **Unsupervised Learning**: Discovers latent topics (K=15) without predefined categories
- **Exclusivity Scoring**: Calculates topic-specific words relative to other topics
- **Comprehensive Evaluation**: Uses multiple quality metrics (coherence, R², log-likelihood)
- **Train-Test Validation**: Evaluates model performance on held-out test data
- **Reproducible**: Uses fixed random seeds for consistent results

## Key Features

- **Balanced Dataset**: Equal representation of positive and negative reviews
- **Robust Text Processing**: Comprehensive cleaning and normalization
- **Multiple ML Algorithms**: Comparison of Naive Bayes, SVM, and Random Forest
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and feature importance
- **Topic Discovery**: LDA-based topic modeling with comprehensive quality metrics
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
- textmineR
- wordcloud2
- ggwordcloud
- Matrix
- textstem
- caTools
- stopwords

## Results

The project generates several outputs:

1. **Processed Datasets**: Clean, balanced review data ready for analysis
2. **Trained Models**: Saved Random Forest, SVM, and LDA topic models for future predictions
3. **Classification Evaluation Visualizations**:
   - ROC curve comparisons
   - Confusion matrix heatmaps
   - Feature importance plots
4. **Topic Modeling Visualizations**:
   - Exclusivity-based word clouds for individual topics
   - Topic labels and key terms with naive labeling
   - Coherence histograms and prevalence scatter plots
   - Topic distribution visualizations for sample reviews

## Model Performance

The project provides comprehensive evaluation metrics for:
- **Classification Models**: Direct comparison of Naive Bayes, SVM, and Random Forest performance
- **Topic Models**: Interpretable topic discovery with 15 topics identified through LDA, with coherence, R², and likelihood metrics

## License

This project is for educational purposes as part of a social media analysis course.

