library(quanteda)
library(mlbench)
library(tm)
library(dplyr)
library(tidyverse)
library(tidytext)
library(Matrix)
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

df_review.tidy.tf <- df_review.tidy.tf %>%
  mutate(
    review_ID = factor(review_ID),
    Word = factor(Word)
  )

cast_dfm(df_review.tidy.tf, review_ID, Word, tf) -> df_review.dfm.tf

convert(df_review.dfm.tf, to="data.frame") -> df_review.tf

stm_data <- quanteda::convert(df_review.dfm.tf, to = "stm")

library(stm)
processed <- stm::prepDocuments(
  documents = stm_data$documents, 
  vocab = stm_data$vocab, 
  meta = stm_data$meta
)


docs <- processed$documents
vocab <- processed$vocab
meta <- processed$meta


#K_results <- stm::searchK(
#     documents = docs, 
#     vocab = vocab, 
#     K = c(3,4,5), 
#     data = meta,
#     init.type = "Spectral"
#)

#plot(K_results)


#K_value <- 4

#model <- stm::stm(
#  documents = docs, 
#  vocab = vocab, 
#  K = K_value,
#  init.type = "Spectral",
#  seed = 888 
#)

model <- readRDS("model/stm_model_K4.rds")

topic_labels <- stm::labelTopics(model, n = 10)
print(topic_labels)


#install.packages("wordcloud2")
#install.packages("ggwordcloud")
library(wordcloud2)
library(ggwordcloud)

# create word cloud baesd on FREX ranking for Topic 1
target_topic <- 1 
N_words <- 50 

topic_labels <- stm::labelTopics(model, n = N_words)
frex_words <- topic_labels$frex[target_topic, ]
topic_betas <- tidy(model, matrix = "beta")
topic_words_df <- topic_betas %>%
  filter(topic == target_topic, 
         term %in% frex_words) %>%
  mutate(
    frex_rank = match(term, frex_words),
    frex_weight = N_words - frex_rank + 1
  ) %>%
  arrange(frex_rank) %>%
  select(word = term, freq = frex_weight)

ggplot(topic_words_df, aes(label = word, size = freq, color = freq)) +
  geom_text_wordcloud_area() +
  scale_size_area(max_size = 15) + 
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(title = paste("Topic", target_topic, "FREX Word Cloud"))

# create word cloud baesd on FREX ranking for Topic 2
target_topic <- 2 
N_words <- 50 

topic_labels <- stm::labelTopics(model, n = N_words)
frex_words <- topic_labels$frex[target_topic, ]
topic_betas <- tidy(model, matrix = "beta")
topic_words_df <- topic_betas %>%
  filter(topic == target_topic, 
         term %in% frex_words) %>%
  mutate(
    frex_rank = match(term, frex_words),
    frex_weight = N_words - frex_rank + 1
  ) %>%
  arrange(frex_rank) %>%
  select(word = term, freq = frex_weight)

ggplot(topic_words_df, aes(label = word, size = freq, color = freq)) +
  geom_text_wordcloud_area() +
  scale_size_area(max_size = 15) + 
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(title = paste("Topic", target_topic, "FREX Word Cloud"))

# create word cloud baesd on FREX ranking for Topic 3
target_topic <- 3 
N_words <- 50 

topic_labels <- stm::labelTopics(model, n = N_words)
frex_words <- topic_labels$frex[target_topic, ]
topic_betas <- tidy(model, matrix = "beta")
topic_words_df <- topic_betas %>%
  filter(topic == target_topic, 
         term %in% frex_words) %>%
  mutate(
    frex_rank = match(term, frex_words),
    frex_weight = N_words - frex_rank + 1
  ) %>%
  arrange(frex_rank) %>%
  select(word = term, freq = frex_weight)

ggplot(topic_words_df, aes(label = word, size = freq, color = freq)) +
  geom_text_wordcloud_area() +
  scale_size_area(max_size = 15) + 
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(title = paste("Topic", target_topic, "FREX Word Cloud"))

# create word cloud baesd on FREX ranking for Topic 4
target_topic <- 4
N_words <- 50 

topic_labels <- stm::labelTopics(model, n = N_words)
frex_words <- topic_labels$frex[target_topic, ]
topic_betas <- tidy(model, matrix = "beta")
topic_words_df <- topic_betas %>%
  filter(topic == target_topic, 
         term %in% frex_words) %>%
  mutate(
    frex_rank = match(term, frex_words),
    frex_weight = N_words - frex_rank + 1
  ) %>%
  arrange(frex_rank) %>%
  select(word = term, freq = frex_weight)

ggplot(topic_words_df, aes(label = word, size = freq, color = freq)) +
  geom_text_wordcloud_area() +
  scale_size_area(max_size = 15) + 
  scale_color_viridis_c() + 
  theme_minimal() +
  labs(title = paste("Topic", target_topic, "FREX Word Cloud"))


# save best model of stm topic model
# saveRDS(model, file = "model/stm_model_K4.rds")
### THE END ###




