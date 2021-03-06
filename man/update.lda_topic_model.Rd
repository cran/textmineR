% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/topic_modeling_core.R
\name{update.lda_topic_model}
\alias{update.lda_topic_model}
\title{Update a Latent Dirichlet Allocation topic model with new data}
\usage{
\method{update}{lda_topic_model}(
  object,
  dtm,
  additional_k = 0,
  iterations = NULL,
  burnin = -1,
  new_alpha = NULL,
  new_beta = NULL,
  optimize_alpha = FALSE,
  calc_likelihood = FALSE,
  calc_coherence = TRUE,
  calc_r2 = FALSE,
  ...
)
}
\arguments{
\item{object}{a fitted object of class \code{lda_topic_model}}

\item{dtm}{A document term matrix or term co-occurrence matrix of class dgCMatrix.}

\item{additional_k}{Integer number of topics to add, defaults to 0.}

\item{iterations}{Integer number of iterations for the Gibbs sampler to run. A
future version may include automatic stopping criteria.}

\item{burnin}{Integer number of burnin iterations. If \code{burnin} is greater than -1,
the resulting "phi" and "theta" matrices are an average over all iterations
greater than \code{burnin}.}

\item{new_alpha}{For now not used. This is the prior for topics over documents
used when updating the model}

\item{new_beta}{For now not used. This is the prior for words over topics
used when updating the model.}

\item{optimize_alpha}{Logical. Do you want to optimize alpha every 10 Gibbs iterations?
Defaults to \code{FALSE}.}

\item{calc_likelihood}{Do you want to calculate the likelihood every 10 Gibbs iterations?
Useful for assessing convergence. Defaults to \code{FALSE}.}

\item{calc_coherence}{Do you want to calculate probabilistic coherence of topics
after the model is trained? Defaults to \code{TRUE}.}

\item{calc_r2}{Do you want to calculate R-squared after the model is trained?
Defaults to \code{FALSE}.}

\item{...}{Other arguments to be passed to \code{\link[textmineR]{TmParallelApply}}}
}
\value{
Returns an S3 object of class c("LDA", "TopicModel").
}
\description{
Update an LDA model with new data using collapsed Gibbs sampling.
}
\examples{
\dontrun{
# load a document term matrix
d1 <- nih_sample_dtm[1:50,]

d2 <- nih_sample_dtm[51:100,]

# fit a model
m <- FitLdaModel(d1, k = 10, 
                 iterations = 200, burnin = 175,
                 optimize_alpha = TRUE, 
                 calc_likelihood = FALSE,
                 calc_coherence = TRUE,
                 calc_r2 = FALSE)

# update an existing model by adding documents
m2 <- update(object = m,
             dtm = rbind(d1, d2),
             iterations = 200,
             burnin = 175)
             
# use an old model as a prior for a new model
m3 <- update(object = m,
             dtm = d2, # new documents only
             iterations = 200,
             burnin = 175)
             
# add topics while updating a model by adding documents
m4 <- update(object = m,
             dtm = rbind(d1, d2),
             additional_k = 3,
             iterations = 200,
             burnin = 175)
             
# add topics to an existing model
m5 <- update(object = m,
             dtm = d1, # this is the old data
             additional_k = 3,
             iterations = 200,
             burnin = 175)

}
}
