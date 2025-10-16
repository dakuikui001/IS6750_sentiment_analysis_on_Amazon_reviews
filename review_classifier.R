# Preprocess reiview csv file with tidytext packages, and convert dataset to Doc-Term-Matrix with tf*idf weighting

library(tidyverse)
library(tidytext)
library(quanteda)
library(mlbench)


### 1. Read in the documents in the review dataset

setwd("/Users/luhui/Documents/R/social_media_analysis/group_project")

df <- read.csv("data/game_reviews.csv", colClasses=c("numeric","factor", "character"))
df_review <- df[,c("label", "text")]

df_review <- df_review %>%
  mutate(review_ID = row_number())
df_review.tidy <- unnest_tokens(df_review, Word, text, token = "words", to_lower = TRUE, drop = TRUE)

# Remove stopwords
data(stop_words)
df_review.tidy <- anti_join (df_review.tidy, stop_words, by = c("Word" = "word"))


### 2. Carry out pre-processing on the review data files
 
# Identify unique words within each sentence with the term frequency (tf)
df_review.tidy %>%
   group_by(review_ID, Word) %>%
   summarize (tf = n()) -> df_review.tidy.tf

# Calculate doc frequency of unique words within the whole dataset, and remove words with df<20
df_review.tidy.tf%>%
  group_by(Word) %>%
  summarize (df = n()) %>%
  filter (df>20) -> df_review.tidy.df

# Remove words with low df
semi_join(df_review.tidy.tf, df_review.tidy.df) -> df_review.tidy.tf

# Calculate tf*idf
bind_tf_idf(df_review.tidy.tf, Word, review_ID, tf) -> df_review.tidy.tf

# Calculate vector length for each sentence (there are long and short sentences)
df_review.tidy.tf %>%
  group_by(review_ID) %>%
  summarize (sentence_len = sqrt(sum(tf_idf^2))) -> df_review.tidy.sentence_len

# Attach the sentence_len value to tf file, and normalize the tf_idf value
df_review.tidy.tf %>%
   left_join(df_review.tidy.sentence_len) %>%
   mutate (tf_idf_norm = tf_idf/sentence_len) -> df_review.tidy.tf

### 3. Join review dataset with the target variable

# Convert tidytext format to document-term matrix
cast_dfm(df_review.tidy.tf, review_ID, Word, tf_idf_norm) -> df_review.dfm.tf

# Convert DFM format to tibble
convert(df_review.dfm.tf, to="data.frame") -> df_review.tf

# Extract the review_ID to merge with the dfm table
group_by(df_review.tidy.tf, review_ID) -> df_review.tidy.tf
summarise(df_review.tidy.tf) -> df_review.tidy.tf.review_ID 

# Bind the review_ID with the dfm table
bind_cols(df_review.tidy.tf.review_ID, df_review.tf) -> df_review.tf

# Join the target variable (label) with dfm table, by the review_ID
inner_join(df_review, df_review.tf, by=c("review_ID" = "review_ID")) -> df_review.tf_label

# Convert the tibbles to data.frame (the data format usually required by machine learning packages)
df_review.tf_label <- as.data.frame(df_review.tf_label)
df_review.tf_label$label <- as.factor(df_review.tf_label$label)

write_csv (df_review.tf_label, path = "data/df_review.tf_label.csv")

library(caret)
library(caTools)

### 4. Partition the dataset into 70% trainingSet and 30% testSet
#df_review.tf_label <- read_csv (file = "data/df_review.tf_label.csv")

# Partition the dataset 
set.seed(1234)
trainIndex <- sample.split( df_review.tf_label$label, SplitRatio = 0.7, group = NULL )

# Create separate training and test set records:
trainingSet <- df_review.tf_label[trainIndex, -c(2:4) ] # remove columns 2 to 4
testSet <- df_review.tf_label[!trainIndex, -c(2:4) ]

#need to identify cases in the trainingSet with 0 variance

## 5. Develop SVM, Naive Bayes using e1071 package and random_forest using Random_forest package

library(e1071)
library(RSNNS)

# load trained rf and svm best model
best_rf_model <- readRDS("model/random_forest_model.rds")
best_svm_model <- readRDS("model/svm_model.rds")

# uniform column name
names(trainingSet) <- make.names(names(trainingSet))
names(testSet) <- make.names(names(testSet))

## a) Develop a na?ve bayes model using the e1071 package:
model_nb2 <- e1071::naiveBayes(label ~ ., data = trainingSet)

# Different machine learning packages use different data formats to represent the output model
# You can check the datatype of the model: class(model_nb2)
# Examine the model by clicking on it in R studio
# You can display the first few lines: head(model_nb2)

# Later you can try other packages with naive bayes modeling: klaR and naivebayes
#	klaR::NaiveBayes ()
#	naivebayes::naive_bayes ()


## b) Develop an SVM model using e1071::svm():

# train time > 10 min!!!
#model_svm2 <- e1071::svm(label ~ ., data = trainingSet, kernel = "radial", probability = TRUE )


# Other packages with SVM:
# e1071 package: svm ()
# klaR package: svmlight ()
# liquidSVM package: mcSVM ()
# kernlab package: a well-known, comprehensive package, but complicated. Need a tutorial/guide


## c): random forest:
library(randomForest)

# train time > 20 min!!!
#model_rf <- randomForest::randomForest(label ~ ., data = trainingSet, ntree = 200, mtry = floor(50), importance = TRUE)


### 6. Using the developed models to make predictions

## a) Apply the models to testSet to generate predicted probabilities of belonging to a class

# Apply the models to the test set to predict label
nb2_pred <- predict(model_nb2, newdata = testSet, type = "class")
nb2_prob <- predict(model_nb2, newdata = testSet, type = "raw")
 
svm2_pred <- predict(best_svm_model, newdata = testSet, probability = TRUE, na.action = na.fail)
svm2_prob <- attr(svm2_pred, "probabilities")

rf_prob <- predict(best_rf_model, testSet, type = "prob")
rf_pred <- predict(best_rf_model, testSet, type = "response")



# According to the manual: predict is a generic function for predictions from the results of various model fitting functions. 
# The function invokes particular methods which depend on the class of the first argument.
# I.e. it calls e1071::predict.naiveBayes, e1071::predict.svm, RSNNS::predict.RSNNS


## b) Evaluate the models: Generate the confusion matrix:

caret::confusionMatrix(data = nb2_pred, reference = testSet$label, mode = "everything")
caret::confusionMatrix(data = svm2_pred, reference = testSet$label, mode = "everything")
caret::confusionMatrix(data = rf_pred, reference = testSet$label, mode = "everything")
# mode can be: "sens_spec", "prec_recall", or "everything"

# Can also use a simple table function to generate a crosstab which functions as a confusion matrix
table(testSet$label, nb2_pred)
table(testSet$label, svm2_pred)
table(testSet$label, rf_pred)


## c) Evaluate the models: Generate evaluation charts using ROCR package

# The prediction () and performance () functions are the core of the analyses in ROCR

# Create the prediction objects:
library(ROCR)
pred_nb2 <- nb2_prob[,"positive"]
pred_svm2 <- svm2_prob[,"positive"]
pred_rf <- rf_prob[,"positive"] 

# Plot the ROC graphs:
library(pROC)
library(ggplot2)

roc_svm <- roc(testSet$label, pred_svm2)
roc_nb <- roc(testSet$label, pred_nb2)
roc_rf <- roc(testSet$label, pred_rf)

ggroc(list(SVM = roc_svm, 'Naive Bayes' = roc_nb, 'Random Forest' = roc_rf), 
      legacy.axes = TRUE) + 
  
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", alpha = 0.7, color = "gray") +
  theme_minimal() +
  labs(title = "ROC Curves Comparison",
       x = "False Positive Rate",
       y = "True Positive Rate",
       color = "Model") + 
  
  scale_color_manual(values = c("SVM" = "red", "Naive Bayes" = "blue", "Random Forest" = "green")) +
  geom_line(lwd = 1) + 
  
  theme(legend.position = c(0.8, 0.2))



# Plot the confusionMatrix heatmap graphs:

library(ggplot2)
library(dplyr) 

model_preds <- list(
  "Random Forest" = rf_pred,
  "SVM" = svm2_pred,
  "Naive Bayes" = nb2_pred
)


testSet$label <- factor(testSet$label)
actual_labels <- testSet$label

all_conf_data <- purrr::map_dfr(names(model_preds), function(model_name) {
  predicted_labels <- factor(model_preds[[model_name]], levels = levels(actual_labels))
  
  conf_matrix <- table(Actual = actual_labels, Predicted = predicted_labels)
  conf_df <- as.data.frame(conf_matrix)
  
  conf_df %>%
    group_by(Actual) %>%
    mutate(
      Model = model_name,
      Proportion = Freq / sum(Freq), 
      Label_Text = paste0(Freq, "\n(", round(Proportion * 100, 1), "%)")
    ) %>%
    ungroup()
})


heatmap_combined <- ggplot(data = all_conf_data, 
                           aes(x = Predicted, y = Actual, fill = Proportion)) + 
  
  geom_tile(color = "black", linewidth = 0.5) +
  
  scale_fill_gradient(low = "white", high = "dodgerblue", 
                      name = "Recall (Row %)") +
  
  geom_text(aes(label = Label_Text), color = "black", size = 4) +
  

  facet_wrap(~ Model, ncol = 3) +
  

  labs(
    title = "Comparison of Confusion Matrices Across Models",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  

  coord_fixed() + 
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(margin = ggplot2::margin(t = 10)),
    axis.title.y = element_text(margin = ggplot2::margin(r = 10)),
    strip.text = element_text(face = "bold"), 
    legend.position = "right"
  )

print(heatmap_combined)


# Plot feature_importance graph of random_forest model:

importance_data <- randomForest::importance(best_rf_model, type = 2) 

importance_df <- as.data.frame(importance_data) %>%
  tibble::rownames_to_column("Feature") %>%
  rename(Importance = MeanDecreaseGini) %>% 

  arrange(desc(Importance)) %>%
  slice_head(n = 20) 

importance_df$Feature <- factor(importance_df$Feature, 
                                levels = importance_df$Feature[order(importance_df$Importance)])

ggplot(importance_df, aes(x = Importance, y = Feature, fill = Importance)) +
  
  geom_bar(stat = "identity") +
  
  geom_text(aes(label = round(Importance, 2)), 
            hjust = -0.1, 
            size = 3) +
  
  labs(
    title = "Top 20 Feature Importance (Mean Decrease Gini)",
    subtitle = "Model: Random Forest",
    x = "Importance Score",
    y = NULL 
  ) +
  
  scale_fill_gradient(low = "lightblue", high = "darkblue") + 
  theme_minimal() +
  theme(
    legend.position = "none", 
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )



# save best model of svm and random_forest
saveRDS(model_svm2, file = "model/svm_model.rds")
saveRDS(model_rf, file = "model/random_forest_model.rds")
### THE END ###




