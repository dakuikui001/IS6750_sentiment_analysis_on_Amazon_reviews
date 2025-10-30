library(tidyverse)
library(tidytext)
library(caTools)
library(caret)
library(e1071)
library(sentimentr)
library(quanteda)
library(quanteda.textmodels)
library(ROCR)
library(jsonlite)
library(textstem)
library(dplyr)
library(wordcloud)
library(ggwordcloud)
library(ggplot2)

setwd("/Users/luhui/Documents/R/social_media_analysis/group_project")

Amazon_data <- read.csv("data/game_reviews_for_topicModeling.csv")
Amazon_data <- Amazon_data %>% 
  dplyr::select(-X)

# Partition the dataset into 50% trainingSet and 50% testSet
# Use sample.split from caTools. 
set.seed(1234)
trainIndex <- sample.split( Amazon_data$Sentiment, SplitRatio = 0.5, group = NULL )

# Create separate training and test set records:
trainingSet <- Amazon_data[trainIndex,] # remove columns not needed in the analysis
testSet <- Amazon_data[!trainIndex,]

# Lemmatize words in the reviewText using textStem::lemmatize_strings
Amazon_data$reviewText_lemma <- lemmatize_strings(Amazon_data$reviewText)


# Topic modeling using textmineR

# Manual: https://cran.r-project.org/web/packages/textmineR/textmineR.pdf

#install.packages("textmineR", dependencies = TRUE)
library(textmineR)


## Code taken from: https://cran.r-project.org/web/packages/textmineR/vignettes/a_start_here.html

# Doesn't need to use Quanteda to process the text
# Create document-term matrix
dtm <- CreateDtm(doc_vec = Amazon_data$reviewText_lemma, # character vector of documents
                 # doc_names = ???, # document names, optional
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(stopwords::stopwords("en"), # stopwords from tm
                                  stopwords::stopwords(source = "smart")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = TRUE, # Turn off status bar for this demo
                 cpus = 2) # by default, this will be the max number of cpus available

head(colnames(dtm))
head(rownames(dtm))

# Calculate DF and IDF for terms
# Not needed for topic modeling
tf_mat <- TermDocFreq(dtm = dtm)
str(tf_mat) 

# list most frequent tokens
head(tf_mat[ order(tf_mat$term_freq, decreasing = TRUE) , ], 10)

# look at the most frequent bigrams
tf_bigrams <- tf_mat[ stringr::str_detect(tf_mat$term, "_") , ]
head(tf_bigrams[ order(tf_bigrams$term_freq, decreasing = TRUE) , ], 10)

# summary of document lengths
doc_lengths <- rowSums(dtm)
summary(doc_lengths)

# remove tokens with DF <20
dtm <- dtm[ , colSums(dtm > 0) > 19 ] # alternatively: dtm[ , tf_mat$term_freq > 3 ]
tf_mat <- tf_mat[ tf_mat$term %in% colnames(dtm) , ]
tf_bigrams <- tf_bigrams[ tf_bigrams$term %in% colnames(dtm) , ]


## Code taken from: https://cran.r-project.org/web/packages/textmineR/vignettes/c_topic_modeling.html

# Split into training & test sets
dtm.train <- dtm[trainIndex == TRUE, ]
dtm.test <- dtm[trainIndex == FALSE, ]


# Fit an LDA model
set.seed(12345)
#LDAmodel2 <- FitLdaModel(dtm = dtm.train, 
#                         k = 15,
#                         iterations = 1000, # use 500 or more 
#                         burnin = 15,
#                         alpha = 0.5,
#                         beta = 0.01,
#                         optimize_alpha = TRUE,
#                         calc_likelihood = TRUE,
#                         calc_coherence = TRUE,
#                         calc_r2 = TRUE,
#                         cpus = 2) 

# Try different k, alpha, beta
LDAmodel2 <- readRDS("model/sdm_model.rds")

# For overall goodness of fit, textmineR has R-squared and log likelihood.
# R-square
LDAmodel2$r2
# log Likelihood (does not consider the prior) 
plot(LDAmodel2$log_likelihood, type = "l")


# Probabilistic coherence, a measure of topic quality
# Probabilistic coherence measures how associated words are in a topic, controlling for statistical independence.
coherence_summary <- summary(LDAmodel2$coherence)
median_coherence <- coherence_summary["Median"]

hist(
  LDAmodel2$coherence,
  col = "#0072B2", 
  border = "white", 
  main = "Histogram of probabilistic coherence", 
  xlab = "Coherence Score", 
  ylab = "Frequency",
  cex.main = 1.3, 
  cex.lab = 1.1   
)

abline(
  v = median_coherence,
  col = "#D55E00", 
  lwd = 2,  
  lty = 2      
)



# Get the prevalence of each topic
# prevalence should be proportional to alpha
plot_data <- data.frame(
  prevalence = LDAmodel2$prevalence,
  alpha = LDAmodel2$alpha
)


ggplot(plot_data, aes(x = prevalence, y = alpha)) +
  
  geom_point(
    size = 3,  
    color = "#0072B2" 
  ) +
  
  geom_smooth(
    method = "loess",  
    se = FALSE, 
    color = "#D55E00", 
    linetype = "dashed"
  ) +
  
  labs(
    x = "Prevalence, %",
    y = "Alpha"
  ) +
  
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16), 
    plot.subtitle = element_text(hjust = 0.5, size = 12),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12)
  )

# List top 20 terms for each topic
# Calculate the most frequent (prevalent) topics in the corpus
# Get some bi-gram topic labels from a naive labeling algorithm
# Combine these together with coherence into a table that summarizes the topic model.
LDAmodel2$top_terms <- GetTopTerms(phi = LDAmodel2$phi, M = 20)

# Get the prevalence of each topic
LDAmodel2$prevalence <- colSums(LDAmodel2$theta) / sum(LDAmodel2$theta) * 100

# textmineR has a naive topic labeling tool based on probable bigrams
LDAmodel2$labels <- LabelTopics(assignments = LDAmodel2$theta > 0.05, 
                                dtm = dtm,
                                M = 1)

# put label&topic¥top_terms together, with coherence into a summary table
LDAmodel2$summary <- data.frame(topic = rownames(LDAmodel2$phi),
                                label = LDAmodel2$labels,
                                coherence = round(LDAmodel2$coherence, 3),
                                prevalence = round(LDAmodel2$prevalence,3),
                                top_terms = apply(LDAmodel2$top_terms, 2, function(x){
                                  paste(x, collapse = ", ")
                                }),
                                stringsAsFactors = FALSE)

LDAmodel2$summary[ order(LDAmodel2$summary$prevalence, decreasing = TRUE) , ][ 1:15 ,]

# Create Word Cloud For each topic
target_topic_id <- 't_14'     # change topic id

phi_matrix <- LDAmodel2$phi

target_phi <- phi_matrix[target_topic_id, ]
other_topics_phi <- phi_matrix[rownames(phi_matrix) != target_topic_id, ]
max_other_phi <- apply(other_topics_phi, 2, max)
epsilon <- 1e-6 
exclusivity_score <- target_phi / (max_other_phi + epsilon)

exclusive_df <- data.frame(
  term = names(exclusivity_score),
  phi_score = target_phi,
  exclusivity = exclusivity_score,
  stringsAsFactors = FALSE
) %>%
  arrange(desc(exclusivity)) %>%
  head(50)

plot_title <- paste0(target_topic_id)

ggplot(
  exclusive_df,
  aes(label = term, size = exclusivity, color = factor(sample.int(10, nrow(exclusive_df), replace = TRUE)))
) +
  geom_text_wordcloud(
    max_size = 8,
    area_max = 100,
    rm_outside = TRUE
  ) +
  scale_size_area(max_size = 10) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16)) +
  labs(title = plot_title) +
  scale_color_discrete(guide = "none")



# Applying the LDA model to a test set
# Prediction with gibbs (preferred but slower)
assignments <- predict(LDAmodel2, dtm.test,
                       method = "gibbs", 
                       iterations = 1000,
                       burnin = 50,
                       cpus = 2)


# visualize the probabilistic distribution of topics across a specific record in test set
# reveal its primary subject matter and topic focus.
barplot(assignments[39,], col = "blue", las = 1, beside = TRUE)

# Comment：
# My favorite NBA K of all time. I play this game way more than any of my other games. 
# Everything about the game is stunning, except maybe LeBron mode. 
# I thought that was kind of ridiculous. Otherwise, completely worth the buy.
  # t_14  0.465
  # t_11  0.2
  # t_10  0.1

barplot(assignments[20,], col = "blue", las = 1, beside = TRUE)
# Comment：
# I bought this for the phone charging capability, but my phone's microUSB slot is on the left side, 
# and it won't line up properly because the controller is too close to the slot. If the arm were longer, 
# this wouldn't be an issue. Also, the arm doesn't rotate enough for my liking. 
# I'd like it to rotate backward more. Fully rotated backward, 
# I feel like I have to hold the controller awkwardly to get the best view of the screen. 
# I'm going to open it up and try to modify the arm or controller plastic so it can rotate further.
 # t_4  0.496
 # t_11 0.334

#saveRDS(LDAmodel2, file = "model/sdm_model.rds")
### END ###
  
  




