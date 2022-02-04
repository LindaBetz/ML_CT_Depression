# ~~~~~~~~~~~~~~~~~~~~~~~~~~     Code for     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                                                            #
#                  The role of childhood abuse and neglect                   #
#                   in depressive affect in adulthood:                       #
#        a machine learning approach in a general population sample          #
#                                                                            #
#     Linda Betz, Marlene Rosen, Raimo K.R. Salokangas, Joseph Kambeitz      #
#                                                                            #
#                                                                            #
#                         Analysis/code by Linda Betz                        #
#                                                                            #
#                                                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# ---------------------- 0: Reproducibility ----------------------

# for reproducibility, we use the "checkpoint" package
# in a new directory, it will *install* those package versions used when the script was written
# these versions are then used to run the script
# to this end, a server with snapshot images of archived package versions needs to be contacted
# for more info visit: https://mran.microsoft.com/documents/rro/reproducibility

library(checkpoint)
checkpoint("2021-06-21",
           R.version = "4.1.0")

# ---------------------- 1: Load packages, functions & data ----------------------

library(tidyverse)
library(coin)
library(patchwork)
library(mlr3)
requireNamespace("mlr3measures")
library(mlr3tuning)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3filters)
library(mlr3viz)
library(NADIA)
library(iml)


# the data sets are available for public use via https://www.icpsr.umich.edu/
load("Biomarker/ICPSR_29282/DS0001/29282-0001-Data.rda") # https://doi.org/10.3886/ICPSR29282.v9
load("Replication/ICPSR_36532-V3/ICPSR_36532/DS0001/36532-0001-Data.rda") # https://doi.org/10.3886/ICPSR36532.v3
load("MIDUS_2/ICPSR_04652/DS0001/04652-0001-Data.rda") # https://doi.org/10.3886/ICPSR04652.v7
load("Replication/ICPSR_36901-V6/ICPSR_36901/DS0001/36901-0001-Data.rda") # https://doi.org/10.3886/ICPSR36901.v6

# load some custom functions moved to extra file for readability of main code
source("Custom_Functions.R")

# ---------------------- 2: Data preparation & sample descriptives ------------------

# ----------- Train Set

pred_data_train <- da29282.0001 %>%
  dplyr::select(
    .,
    matches(
      "M2ID|B4ZSITE|B4ZCOMPM|B4ZCOMPY|B1PRSEX|^B4ZAGE$|^B4QCT_[PA|EA|SA|EN|PN]|^B4QCESDDA$"
    )
  ) %>%
  dplyr::filter(B4ZAGE <= 60) %>% # exclude those older than 60 years
  mutate(outcome = B4QCESDDA,
         collection_date = as.Date(paste0(
           "01", "-", str_extract(B4ZCOMPM, "[0-9][0-9]"), "-", B4ZCOMPY
         ), format = "%d-%m-%Y")) %>%
  dplyr::select(-c(M2ID, B4QCESDDA, B4ZCOMPM, B4ZCOMPY)) %>%
  dplyr::select(sort(names(.))) %>%
  mutate(across(matches("B4Q9"), ~ ordered(.))) %>% #
  mutate(across(matches("B1PRSEX|B1PF7A"), ~ as.factor(.))) %>%
  filter(!is.na(outcome)) %>%
  dplyr::select(outcome, collection_date, everything()) %>% # to ensure correct ordering of vars
  filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25)


nrow(pred_data_train) # 769 people in train set

# sex distribution
table(pred_data_train$B1PRSEX) / nrow(pred_data_train)
# (1) MALE (2) FEMALE
# 0.4278283  0.5721717

# data collection sites
table(pred_data_train$B4ZSITE) / nrow(pred_data_train)
# (1) UCLA         (2) UW (3) GEORGETOWN
# 0.3237971      0.4577373      0.2184655

# data collection period
min(pred_data_train$collection_date) # start of data collection: 2004-07-01
max(pred_data_train$collection_date) # end of data collection: 2009-05-01

# other continuous variables
pred_data_train %>%
  summarize(across(where(is.numeric), list(
    mean = ~ mean(.x, na.rm = T),
    sd = ~ sd(.x, na.rm = T)
  ))) %>% round(., 1)

# ----------- Test Set

pred_data_test <- da36901.0001 %>%
  dplyr::select(
    .,
    matches(
      "MRID|RA4ZCOMPM|RA4ZCOMPY|RA4ZSITE|RA1PRSEX|^RA4ZAGE$|^RA4QCT_[PA|EA|SA|EN|PN]|^RA4QCESDDA$"
    )
  ) %>%
  dplyr::filter(RA4ZAGE >= 35 &
                  RA4ZAGE <= 60) %>% # same age range
  mutate(outcome = RA4QCESDDA,
         collection_date = as.Date(paste0(
           "01", "-", str_extract(RA4ZCOMPM, "[0-9][0-9]"), "-", RA4ZCOMPY
         ), format = "%d-%m-%Y")) %>%
  dplyr::select(-c(MRID, RA4QCESDDA, RA4ZCOMPM, RA4ZCOMPY)) %>%
  dplyr::select(sort(names(.))) %>% # to ensure correct naming later on
  mutate(across(matches("RA1PRSEX|RA1PF7A"), ~ as.factor(toupper(.)))) %>% # for correspondence with training data set
  dplyr::select(sort(names(.))) %>%
  mutate(across(matches("RA4Q9"), ~ ordered(toupper(.)))) %>%  # to ensure correct collapsing of factor levels
  mutate(across(matches("RA4Q9_"), ~ as.numeric(.))) %>%
  filter(!is.na(outcome)) %>%
  dplyr::select(outcome, collection_date, everything()) %>% # to ensure correct ordering of vars
  filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25)


nrow(pred_data_test) # 466 people in test set

# sex distribution
table(pred_data_test$RA1PRSEX) / nrow(pred_data_test)
# (1) MALE (2) FEMALE
# 0.4206009  0.5793991

# data collection sites
table(pred_data_test$RA4ZSITE) / nrow(pred_data_test)
# (1) UCLA         (2) UW (3) Georgetown
# 0.3347639      0.4313305      0.2339056

# data collection period
min(pred_data_test$collection_date) # start of data collection: 2012-10-01
max(pred_data_test$collection_date)  # end of data collection: 2016-08-01

# other continuous variables
pred_data_test %>%
  summarize(across(where(is.numeric), list(
    mean = ~ mean(.x, na.rm = T),
    sd = ~ sd(.x, na.rm = T)
  ))) %>% round(., 1)


# ---------------------- 3: Comparison: Train & Test Sample ---------------------

colnames(pred_data_test) <- colnames(pred_data_train)

# merge data for comparison
merged_data <- pred_data_train %>%
  mutate(sample = "training") %>%
  bind_rows(pred_data_test %>% mutate(sample = "test"))

# continuous variables
set.seed(123)
merged_data %>%
  dplyr::select(outcome,
                B4QCT_EA,
                B4QCT_EN,
                B4QCT_PA,
                B4QCT_PN,
                B4QCT_SA,
                B4ZAGE) %>%
  map(
    .,
    ~ oneway_test(
      as.numeric(.) ~ factor(merged_data$sample, levels = c("training", "test")),
      data = merged_data,
      distribution = "approximate"
    )
  )

# categorical variables
set.seed(123)
merged_data %>%
  dplyr::select(B1PRSEX,
                B4ZSITE) %>%
  map(.,
      ~ chisq_test(
        as.factor(.) ~ factor(merged_data$sample, levels = c("training", "test")),
        data = merged_data,
        distribution = "approximate"
      ))

# ---------------------- 4: Train Machine Learning Models ----------------------

# ----------- CT Domain Scores
# train model --> takes 1-2 hours to run w/o parallelization
set.seed(234)
ct_domains_model <- train_model(input = "ct_domains")

# apply to test data set
ct_domains_test <-
  apply_model(ct_domains_model, input = "ct_domains")

# extract aggregated performance
ct_domains_test$score(msr("regr.rsq")) # regr.rsq 0.06430464 


# ----------- CT Individual Items
# train model --> takes 1-2 hours to run w/o parallelization
set.seed(234)
ct_items_model <-
  train_model(input = "ct_items")

# apply to test data set
ct_items_test <-
  apply_model(ct_items_model, input = "ct_items")

# extract aggregated performance
ct_items_test$score(msr("regr.rsq")) # regr.rsq 0.07582186 


# ---------------------- 5: Compute Permutation Importance ----------------------

# ----------- CT Domain Scores
set.seed(234)
ct_domains_perm <-
  permutation_importance(ct_domains_model, input = "ct_domains")


plot_permutation_importance(ct_domains_perm,  input = "ct_domains") +
  theme(legend.position = "none") +
  plot_annotation("Feature Importance: Domains")

ggsave(
  "Figure_2.pdf",
  device = "pdf",
  width = 17 / 1.5,
  height =  7.75,
  units = "cm"
)

# ----------- CT Individual Items
set.seed(234)
ct_items_perm <-
  permutation_importance(ct_items_model, input = "ct_items")

plot_permutation_importance(ct_items_perm,  input = "ct_items") +
  plot_annotation("Feature Importance: Items")


ggsave(
  "Figure_3.pdf",
  device = "pdf",
  width = 17,
  height = 15.5,
  units = "cm"
)


# ---------------------- 6: Generate ALE (plots) -----------------------

# ----------- CT Domain Scores

ct_domains_ale <- ale(ct_domains_model,  input = "ct_domains")

ale_plot <- list()
ale_list <- ct_domains_ale

ale_plot[[1]] <-
  ale_list[[2]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Emotional Abuse") +
  ggtitle("") +
  geom_line(color = "#4DBBD5FF", size = 1)

ale_plot[[2]] <-
  ale_list[[3]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Emotional Neglect") +
  ggtitle("") +
  geom_line(color = "#E64B35FF", size = 1)

ale_plot[[3]] <-
  ale_list[[4]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Physical Abuse") +
  geom_line(color = "#8491B4FF", size = 1)

ale_plot[[4]] <-
  ale_list[[5]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Physical Neglect") +
  ggtitle("") +
  geom_line(color = "#F39B7FFF", size = 1)

ale_plot[[5]] <-
  ale_list[[6]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Sexual Abuse") +
  ggtitle("") +
  geom_line(color = "#3C5488FF", size = 1)

ale_plot[[1]] + ale_plot[[2]] + ale_plot[[3]] + ale_plot[[4]] + ale_plot[[5]] +
  plot_annotation(
    title = 'Accumulated Local Effects for Depressive Affect: Domains')

ggsave(
  "Figure_4.pdf",
  device = "pdf",
  width = 21,
  height = 13,
  units = "cm"
)


# ----------- CT Individual Items

ct_items_ale <- ale(ct_items_model,  input = "ct_items")

# extract 6 most important features - we'll show ALE plots for them
ct_items_perm$results %>%
  mutate(importance = -importance) %>%
  top_n(6, wt = importance) %>%
  arrange(-importance)


ale_plot_items <- list()
ale_list_items <- ct_items_ale

ale_plot_items[[1]] <-
  ale_list_items[[19]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Felt family member hated me") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#4DBBD5FF")

ale_plot_items[[2]] <-
  ale_list_items[[25]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Believe I was emotionally abused") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#4DBBD5FF")

ale_plot_items[[3]] <-
  ale_list_items[[22]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Other threaten harm if no sexual acts") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#3C5488FF")

ale_plot_items[[4]] <-
  ale_list_items[[16]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Family said hurtful things to me") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#4DBBD5FF")


ale_plot_items[[5]] <-
  ale_list_items[[11]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("Thought parents wished I wasn't born") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#4DBBD5FF")


ale_plot_items[[6]] <-
  ale_list_items[[26]]$plot() +
  my_theme +
  scale_y_continuous("", limits = limits) +
  xlab("I was taken to doctor if needed") +
  ggtitle("") +
  geom_bar(stat = "identity", fill = "#F39B7FFF")


ale_plot_items[[1]] + 
  ale_plot_items[[2]] + 
  ale_plot_items[[3]] + 
  ale_plot_items[[4]] + 
  ale_plot_items[[5]] + 
  ale_plot_items[[6]] +
  plot_annotation(
    title = 'Accumulated Local Effects for Depressive Affect: Items')


ggsave(
  "Figure_5.pdf",
  device = "pdf",
  width = 21.65,
  height = 13,
  units = "cm"
)


# ---------------------- 7: Assess the role of potential confounds ------------------

# following Dinga et al., 2020 (https://doi.org/10.1101/2020.08.17.255034)

# ----------- CT Domain Scores

domain_predictions <-
  cbind.data.frame(prediction = ct_domains_test$response,
                   pred_data_test %>% dplyr::select("outcome",
                                                    "RA4ZAGE",   "RA4ZSITE"))
# fit models we will compare
# 1. model with only confounds as predictors
m_conf <-
  lm(outcome ~  RA4ZSITE + RA4ZAGE, data = domain_predictions)

# 2. model with only ML predictions as predictors
m_pred <- lm(outcome ~ prediction, data = domain_predictions)

# 3. model with both ML predictions and confounds as covariates
m_conf_pred <- lm(outcome ~   RA4ZSITE + RA4ZAGE + prediction,
                  data = domain_predictions)

decompose_r2(m_conf, m_pred, m_conf_pred)


# ----------- CT Individual Items

item_predictions <-
  cbind.data.frame(prediction = ct_items_test$response,
                   pred_data_test %>% dplyr::select("outcome",
                                             "RA4ZAGE",   "RA4ZSITE"))

# fit models we will compare
# 1. model with only confounds as predictors
m_conf <-
  lm(outcome ~  RA4ZAGE + RA4ZSITE, data = item_predictions)

# 2. model with only ML predictions as predictors
m_pred <- lm(outcome ~ prediction, data = item_predictions)

# 3. model with both ML predictions and confounds as covariates
m_conf_pred <- lm(outcome ~  RA4ZAGE + RA4ZSITE + prediction,
                  data = item_predictions)

decompose_r2(m_conf, m_pred, m_conf_pred)
