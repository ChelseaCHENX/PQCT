library(table1)
library(dplyr)
# library(lme4)
# library(purrr)
# library(broom)
library(tidyr)
# library(readxl)
# library(writexl)
# library(gtsummary)
# library(psych)
# library(corrplot)
# library(ggplot2)
# library(car)
# library(dagitty)
# library(lavaan)
# library(glue)

pvalueANOVA <- function(x, ...) {
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  
  if (is.numeric(y)) {
    # For numeric variables, perform a standard 2-sample t-test
    ano <- aov(y ~ g)
    p <- summary(ano)[[1]][[5]][1]
    
  } else {
    # For categorical variables, perform a chi-squared test of independence
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-value, using an HTML entity for the less-than sign.
  # The initial empty string places the output on the line below the variable label.
  c("", sub("<", "&lt;", format.pval(p, digits=3, eps=0.001)))
}

pvalueMannWhitneyU <- function(x, ...) {
  # Construct vectors of data y, and groups (strata) g
  y <- unlist(x)
  g <- factor(rep(1:length(x), times=sapply(x, length)))
  if (is.numeric(y)) {
    # For numeric variables, perform a standard 2-sample t-test
    p <- wilcox.test(y ~ g)$p.value
    
  } else {
    # For categorical variables, perform a chi-squared test of independence
    p <- chisq.test(table(y, g))$p.value
  }
  # Format the p-value, using an HTML entity for the less-than sign.
  # The initial empty string places the output on the line below the variable label.
  c("", sub("<", "&lt;", format.pval(p, digits=3, eps=0.001)))
}