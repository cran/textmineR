## ----setup, include = FALSE----------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>", warning = FALSE
)

## ------------------------------------------------------------------------
library(textmineR)

# load movie_review dataset from text2vec
data(movie_review, package = "text2vec")

str(movie_review)

# let's take a sample so the demo will run quickly
# note: textmineR is generally quite scaleable, depending on your system
set.seed(123)
s <- sample(1:nrow(movie_review), 500)

movie_review <- movie_review[ s , ]

# create a document term matrix 
dtm <- CreateDtm(doc_vec = movie_review$review, # character vector of documents
                 doc_names = movie_review$id, # document names
                 ngram_window = c(1, 2), # minimum and maximum n-gram length
                 stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                  tm::stopwords("SMART")), # this is the default value
                 lower = TRUE, # lowercase - this is the default value
                 remove_punctuation = TRUE, # punctuation - this is the default
                 remove_numbers = TRUE, # numbers - this is the default
                 verbose = FALSE, # Turn off status bar for this demo
                 cpus = 2) # default is all available cpus on the system

## ------------------------------------------------------------------------

# Fit a Latent Dirichlet Allocation model
# note the number of topics is arbitrary here
# see extensions for more info
model <- FitLdaModel(dtm = dtm, 
                     k = 100, 
                     iterations = 200, # i recommend a larger value, 500 or more
                     alpha = 0.1, # this is the default value
                     beta = 0.05, # this is the default value
                     cpus = 2) 


## ------------------------------------------------------------------------
# two matrices: 
# theta = P(topic | document)
# phi = P(word | topic)
str(model)


## ------------------------------------------------------------------------

# R-squared 
# - only works for probabilistic models like LDA and CTM
model$r2 <- CalcTopicModelR2(dtm = dtm, 
                             phi = model$phi,
                             theta = model$theta,
                             cpus = 2)

model$r2

# log Likelihood (does not consider the prior) 
# - only works for probabilistic models like LDA and CTM
model$ll <- CalcLikelihood(dtm = dtm, 
                           phi = model$phi, 
                           theta = model$theta,
                           cpus = 2)

model$ll

## ----fig.width = 7.5, fig.height = 4-------------------------------------
# probabilistic coherence, a measure of topic quality
# this measure can be used with any topic model, not just probabilistic ones
model$coherence <- CalcProbCoherence(phi = model$phi, dtm = dtm, M = 5)

summary(model$coherence)

hist(model$coherence, 
     col= "blue", 
     main = "Histogram of probabilistic coherence")

## ------------------------------------------------------------------------
# Get the top terms of each topic
model$top_terms <- GetTopTerms(phi = model$phi, M = 5)

## ----eval = FALSE--------------------------------------------------------
#  head(t(model$top_terms)

## ---- echo = FALSE-------------------------------------------------------
knitr::kable(head(t(model$top_terms)), 
             col.names = rep("", nrow(model$top_terms)))

## ------------------------------------------------------------------------
# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
model$prevalence <- colSums(model$theta) / sum(model$theta) * 100

# textmineR has a naive topic labeling tool based on probable bigrams
model$labels <- LabelTopics(assignments = model$theta > 0.05, 
                            dtm = dtm,
                            M = 1)

head(model$labels)

# put them together, with coherence into a summary table
model$summary <- data.frame(topic = rownames(model$phi),
                            label = model$labels,
                            coherence = round(model$coherence, 3),
                            prevalence = round(model$prevalence,3),
                            top_terms = apply(model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)


## ----eval = FALSE--------------------------------------------------------
#  model$summary[ order(model$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ]

## ----echo = FALSE--------------------------------------------------------
knitr::kable(model$summary[ order(model$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ], caption = "Summary of 10 most prevalent topics")


## ------------------------------------------------------------------------

# first get a prediction matrix, phi is P(word | topic)
# we need P(topic | word), or "phi_prime"
model$phi_prime <- CalcPhiPrime(phi = model$phi,
                                theta = model$theta)

# set up the assignments matrix and a simple dot product gives us predictions
assignments <- dtm / rowSums(dtm)

assignments <- assignments %*% t(model$phi_prime)

assignments <- as.matrix(assignments) # convert to regular R dense matrix

## ----fig.width = 7.5, fig.height = 4-------------------------------------
# compare the "fit" assignments to the predicted ones
barplot(rbind(model$theta[ rownames(dtm)[ 1 ] , ],
              assignments[ rownames(dtm)[ 1 ] , ]), 
        las = 2,
        main = "Comparing topic assignments",
        beside = TRUE,
        col = c("red", "blue"))

legend("topleft", 
       legend = c("Bayesian (during fitting)", "Frequentist (predicted)"),
       fill = c("red", "blue"))


## ------------------------------------------------------------------------

# get a tf-idf matrix
tf_sample <- TermDocFreq(dtm)

tf_sample$idf[ is.infinite(tf_sample$idf) ] <- 0 # fix idf for missing words

tf_idf <- t(dtm / rowSums(dtm)) * tf_sample$idf

tf_idf <- t(tf_idf)

# Fit a Latent Semantic Analysis model
# note the number of topics is arbitrary here
# see extensions for more info
lsa_model <- FitLsaModel(dtm = tf_idf, 
                     k = 100)

# three objects: 
# theta = distribution of topics over documents
# phi = distribution of words over topics
# sv = a vector of singular values created with SVD
str(lsa_model)

## ----fig.width = 7.5, fig.height = 4-------------------------------------
# probabilistic coherence, a measure of topic quality
# - can be used with any topic lsa_model, e.g. LSA
lsa_model$coherence <- CalcProbCoherence(phi = lsa_model$phi, dtm = dtm, M = 5)

summary(lsa_model$coherence)

hist(lsa_model$coherence, col= "blue")

# Get the top terms of each topic
lsa_model$top_terms <- GetTopTerms(phi = lsa_model$phi, M = 5)

## ----eval = FALSE--------------------------------------------------------
#  head(t(lsa_model$top_terms))

## ----echo = FALSE--------------------------------------------------------
knitr::kable(head(t(lsa_model$top_terms)), 
             col.names = rep("", nrow(lsa_model$top_terms)))

## ------------------------------------------------------------------------

# Get the prevalence of each topic
# You can make this discrete by applying a threshold, say 0.05, for
# topics in/out of docuemnts. 
lsa_model$prevalence <- colSums(lsa_model$theta) / sum(lsa_model$theta) * 100

# textmineR has a naive topic labeling tool based on probable bigrams
lsa_model$labels <- LabelTopics(assignments = lsa_model$theta > 0.05, 
                            dtm = dtm,
                            M = 1)

## ----eval = FALSE--------------------------------------------------------
#  head(lsa_model$labels)

## ----echo = FALSE--------------------------------------------------------
knitr::kable(head(lsa_model$labels))


## ------------------------------------------------------------------------

# put them together, with coherence into a summary table
lsa_model$summary <- data.frame(topic = rownames(lsa_model$phi),
                            label = lsa_model$labels,
                            coherence = round(lsa_model$coherence, 3),
                            prevalence = round(lsa_model$prevalence,3),
                            top_terms = apply(lsa_model$top_terms, 2, function(x){
                              paste(x, collapse = ", ")
                            }),
                            stringsAsFactors = FALSE)

## ----eval = FALSE--------------------------------------------------------
#  lsa_model$summary[ order(lsa_model$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ]

## ----echo = FALSE--------------------------------------------------------
knitr::kable(lsa_model$summary[ order(lsa_model$summary$prevalence, decreasing = TRUE) , ][ 1:10 , ], caption = "Summary of 10 most prevalent LSA topics")


## ------------------------------------------------------------------------
# Get topic predictions for all 5,000 documents

# first get a prediction matrix,
lsa_model$phi_prime <- diag(lsa_model$sv) %*% lsa_model$phi

lsa_model$phi_prime <- t(MASS::ginv(lsa_model$phi_prime))

# set up the assignments matrix and a simple dot product gives us predictions
lsa_assignments <- t(dtm) * tf_sample$idf

lsa_assignments <- t(lsa_assignments)

lsa_assignments <- lsa_assignments %*% t(lsa_model$phi_prime)

lsa_assignments <- as.matrix(lsa_assignments) # convert to regular R dense matrix



## ----fig.width = 7.5, fig.height = 4-------------------------------------
# compare the "fit" assignments to the predicted ones
barplot(rbind(lsa_model$theta[ rownames(dtm)[ 1 ] , ],
              lsa_assignments[ rownames(dtm)[ 1 ] , ]), 
        las = 2,
        main = "Comparing topic assignments in LSA",
        beside = TRUE,
        col = c("red", "blue"))

legend("topleft", 
       legend = c("During fitting", "Predicted"),
       fill = c("red", "blue"))


## ----fig.width = 7.5, fig.height = 4-------------------------------------
# load a sample DTM
data(nih_sample_dtm)

# choose a range of k 
# - here, the range runs into the corpus size. Not recommended for large corpora!
k_list <- seq(5, 95, by = 5)

# you may want toset up a temporary directory to store fit models so you get 
# partial results if the process fails or times out. This is a trivial example, 
# but with a decent sized corpus, the procedure can take hours or days, 
# depending on the size of the data and complexity of the model.
# I suggest using the digest package to create a hash so that it's obvious this 
# is a temporary directory
model_dir <- paste0("models_", digest::digest(colnames(nih_sample_dtm), algo = "sha1"))

# Fit a bunch of LDA models
# even on this trivial corpus, it will a bit of time to fit all of these models
model_list <- TmParallelApply(X = k_list, FUN = function(k){

  m <- FitLdaModel(dtm = nih_sample_dtm, 
                   k = k, 
                   iterations = 200, 
                   cpus = 1)
  m$k <- k
  m$coherence <- CalcProbCoherence(phi = m$phi, 
                                   dtm = nih_sample_dtm, 
                                   M = 5)
  m
}, export=c("nih_sample_dtm"), # export only needed for Windows machines
cpus = 2) 

# Get average coherence for each model
coherence_mat <- data.frame(k = sapply(model_list, function(x) nrow(x$phi)), 
                            coherence = sapply(model_list, function(x) mean(x$coherence)), 
                            stringsAsFactors = FALSE)


# Plot the result
# On larger (~1,000 or greater documents) corpora, you will usually get a clear peak
plot(coherence_mat, type = "o")
    


