# ~~~~~~~~~~~~~~~~~~~~~~~~~~     Code for     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                                                                            #
#         Disentangling the impact of childhood abuse and neglect            #
#                 on depressive affect in adulthood:                         #
#      a machine learning approach in a general population sample.           #
#                                                                            #
#     Linda Betz, Marlene Rosen, Raimo K.R. Salokangas, Joseph Kambeitz      #
#                                                                            #
#                                                                            #
#                         Analysis/code by Linda Betz                        #
#                                                                            #
#                                                                            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Custom functions used in "Main_Analysis.R"

train_model <-
  function(input = c("ct_items",
                     "ct_domains")) {
    input_string <-
      case_when(
        input == "ct_items" ~ "M2ID|SAMPLMAJ|B1PRSEX|^B4ZAGE$|B4Q9[A-I]|B4Q9[K-O]|B4Q9[Q-U]|B4Q9[W-Z]|^B4QCESDDA$",
        input == "ct_domains" ~ "M2ID|SAMPLMAJ|B1PRSEX|^B4ZAGE$|^B4QCT_[PA|EA|SA|EN|PN]|^B4QCESDDA$",
        TRUE ~  "M2ID|SAMPLMAJ|B1PRSEX|^B4ZAGE$|^B4QCESDDA$"
      )
    
    pred_data_train <- da29282.0001 %>%
      dplyr::select(.,
                    matches(input_string)) %>%
      dplyr::filter(B4ZAGE <= 60) %>% # exclude those older than 60 years
      mutate(outcome = B4QCESDDA) %>%
      dplyr::select(-c(M2ID, SAMPLMAJ, B4QCESDDA, B4ZAGE)) %>%
      dplyr::select(sort(names(.))) %>% # to ensure correct order for correspondence between train & test ds
      mutate(across(matches("B4Q9"), ~ ordered(.))) %>% # to ensure correct collapsing of CTQ items' factor levels
      mutate(across(matches("B4Q9_"), ~ as.numeric(.))) %>%
      mutate(across(matches("B1PRSEX"), ~ as.factor(.))) %>%
      dplyr::filter(!is.na(outcome)) %>%
      dplyr::filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25)
    
    
    # set up ML
    task_train <- TaskRegr$new("midus",
                               pred_data_train,
                               target = "outcome")
    
    # setting up learner (random forest)
    learner <- lrn(
      "regr.ranger",
      num.trees = 1000,
      importance = "permutation",
      replace = FALSE
    )
    
    unordered_factor_vars <-
      task_train$feature_names[str_detect(task_train$feature_names, "B1PRSEX")]
    
    ordered_factor_vars <-
      task_train$feature_names[str_detect(task_train$feature_names, "B4Q9[A-Z]")]
    
    # construct a graph learner to implement pre-processing steps within resampling
    graph_lrn <- GraphLearner$new(
      po("collapsefactors", target_level_count = 2) %>>%
        po("fixfactors") %>>%
        po(
          "colapply",
          id = "fact_to_num",
          applicator = as.numeric,
          affect_columns = selector_name(c(
            unordered_factor_vars, ordered_factor_vars
          ))
        ) %>>%
        po("VIM_kNN_imputation") %>>%
        po(
          "colapply",
          id = "num_to_ordered_fact",
          applicator = function(x)
            ordered(as.integer(round(x, 0L))),
          affect_columns = selector_name(ordered_factor_vars)
        )  %>>%
        po(
          "colapply",
          id = "num_to_unordered_fact",
          applicator = function(x)
            as.factor(as.integer(round(x, 0L))),
          affect_columns = selector_name(unordered_factor_vars)
        )  %>>%
        po(learner)
    )
    
    resampling <- rsmp("repeated_cv", folds = 5, repeats = 5)
    
    # performance measure
    measure <- msr("regr.rsq")
    
    # define search space of hyperparameters to tune
    search_space <- paradox::ParamSet$new(
      params = list(
        paradox::ParamInt$new(
          "regr.ranger.mtry",
          lower = 1,
          upper = length(task_train$feature_names),
          # number of features
          default = length(task_train$feature_names)
        ),
        paradox::ParamDbl$new(
          "regr.ranger.sample.fraction",
          lower = 0.2,
          upper = 0.9,
          default = 0.66
        ),
        paradox::ParamInt$new(
          "regr.ranger.min.node.size",
          lower = 1,
          upper = ceiling(task_train$nrow * 0.2)
        ),
        paradox::ParamDbl$new(
          "regr.ranger.regularization.factor",
          lower = 0,
          upper = 1,
          default = 0.5
        ),
        paradox::ParamDbl$new(
          "collapsefactors.no_collapse_above_prevalence",
          lower = 0.01,
          upper = 0.1,
          default = 0.025
        ),
        paradox::ParamInt$new(
          "impute_VIM_kNN_B.k",
          lower = 1,
          upper = 20,
          default = 10
        )
      )
    )
    
    # define optimization
    terminator <- trm("evals", n_evals = 100)
    tuner <- mlr3tuning::tnr("random_search")
    
    
    instance <- TuningInstanceSingleCrit$new(
      task = task_train,
      learner = graph_lrn,
      resampling = resampling,
      measure = measure,
      search_space = search_space,
      terminator = terminator,
      store_models = FALSE
    )
    
    # disable very verbose default output
    # lgr::get_logger("mlr3")$set_threshold("warn")
    # lgr::get_logger("bbotk")$set_threshold("warn")
    
    future::plan("multiprocess", workers = future::availableCores() - 1)
    
    tuner$optimize(instance)
    
    
    # get "final model"
    graph_lrn$param_set$values <-
      instance$result_learner_param_vals
    
    
    graph_lrn$train(task_train)
    
    return(graph_lrn)
  }


# test model
apply_model <-
  function(trained_model,
           input = c("ct_items",
                     "ct_domains")) {
    input_string <-
      case_when(
        input == "ct_items" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|RA4Q9[A-I]|RA4Q9[K-O]|RA4Q9[Q-U]|RA4Q9[W-Z]|^RA4QCESDDA$",
        input == "ct_domains" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|^RA4QCT_[PA|EA|SA|EN|PN]|^RA4QCESDDA$",
        TRUE ~  "MRID|SAMPLMAJ|^RA4ZAGE$|RA1PRSEX|^RA4QCESDDA$"
      )
    
    pred_data_test <- da36901.0001 %>%
      dplyr::select(.,
                    matches(input_string)) %>%
      dplyr::filter(RA4ZAGE >= 35 &
                      RA4ZAGE <= 60) %>% # same age limit
      mutate(outcome = RA4QCESDDA) %>%
      dplyr::select(-c(MRID, SAMPLMAJ, RA4QCESDDA, RA4ZAGE)) %>%
      dplyr::select(sort(names(.))) %>% # to ensure correct naming later on
      mutate(across(matches("RA1PRSEX"), ~ as.factor(toupper(.)))) %>% # for correspondence with training data set
      dplyr::select(sort(names(.))) %>%
      mutate(across(matches("RA4Q9"), ~ ordered(toupper(.)))) %>%  # to ensure correct collapsing of factor levels
      mutate(across(matches("RA4Q9_"), ~ as.numeric(.))) %>%  # to ensure correct collapsing of factor levels
      dplyr::filter(!is.na(outcome)) %>%
      dplyr::filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25) %>%
      dplyr::select(outcome, everything()) # to ensure correct ordering of vars
    
    colnames(pred_data_test) <-
      c("outcome", sort(
        names(
          trained_model$model$regr.ranger$model$variable.importance
        )
      )) # to ensure correct application of the model
    
    prediction_test  <-
      trained_model$predict_newdata(pred_data_test)
    return(prediction_test)
    
  }


permutation_importance <- function(trained_model, input) {
  input_string <-
    case_when(
      input == "ct_items" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|RA4Q9[A-I]|RA4Q9[K-O]|RA4Q9[Q-U]|RA4Q9[W-Z]|^RA4QCESDDA$",
      input == "ct_domains" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|^RA4QCT_[PA|EA|SA|EN|PN]|^RA4QCESDDA$",
      TRUE ~  "MRID|SAMPLMAJ|^RA4ZAGE$|RA1PRSEX|^RA4QCESDDA$"
    )
  
  pred_data_test <- da36901.0001 %>%
    dplyr::select(.,
                  matches(input_string)) %>%
    dplyr::filter(RA4ZAGE >= 35 &
                    RA4ZAGE <= 60) %>% # same age limit as in training data set
    mutate(outcome = RA4QCESDDA) %>%
    dplyr::select(-c(MRID, SAMPLMAJ, RA4QCESDDA, RA4ZAGE)) %>%
    dplyr::select(sort(names(.))) %>% # to ensure correct naming later on
    mutate(across(matches("RA1PRSEX"), ~ as.factor(toupper(.)))) %>% # for correspondence with training data set
    dplyr::select(sort(names(.))) %>%
    mutate(across(matches("RA4Q9"), ~ ordered(toupper(.)))) %>%  # to ensure correct collapsing of factor levels
    mutate(across(matches("RA4Q9_"), ~ as.numeric(.))) %>%  # to ensure correct collapsing of factor levels
    filter(!is.na(outcome)) %>%
    filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25) %>%
    dplyr::select(outcome, everything()) # to ensure correct ordering of vars
  
  colnames(pred_data_test) <-
    c("outcome", sort(
      names(
        trained_model$model$regr.ranger$model$variable.importance
      )
    )) # to ensure correct application of the model
  
  
  task_test <- TaskRegr$new("midus",
                            pred_data_test,
                            target = "outcome")
  
  # preprocessing of test data with parameters learned on train data
  preproc_graph_test <-
    trained_model$graph$pipeops$collapsefactors %>>%
    trained_model$graph$pipeops$fixfactors  %>>%
    trained_model$graph$pipeops$fact_to_num %>>%
    trained_model$graph$pipeops$impute_VIM_kNN_B  %>>%
    trained_model$graph$pipeops$num_to_ordered_fact %>>%
    trained_model$graph$pipeops$num_to_unordered_fact
  
  preproc_graph_test$train(task_test)
  
  preproc_graph_test <-
    preproc_graph_test$predict(task_test)
  
  preproc_data_test <-
    as_tibble(preproc_graph_test$num_to_unordered_fact$data())
  
  pfun <-
    function(object, newdata)
      predict(object, data = newdata)$predictions
  
  predictor_dep_test <-
    Predictor$new(
      trained_model$model$regr.ranger$model,
      data = preproc_data_test %>% dplyr::select(-outcome),
      y = preproc_data_test %>% dplyr::select(outcome),
      predict.fun = pfun
    )
  
  rsq_error <- function(actual, predicted)
    mlr3measures::rsq(actual, predicted)
  
  future::plan("multiprocess", workers = future::availableCores() - 1)
  permutation_importance <-
    FeatureImp$new(
      predictor_dep_test,
      loss = rsq_error,
      compare = "difference",
      n.repetitions = 1000
    )
  
  return(permutation_importance)
  
}


plot_permutation_importance <-
  function(permutation_importance, input) {
    demographics <- c("Sex")
    
    ct_items <-  c(
      "Didn't have enough to eat",
      # a
      "Believe I was sexually abused",
      # aa
      "Knew someone was there for me",
      # b
      "Family source of strength-support",
      # bb
      "Family called me names",
      # c
      "Parents drunk-high, care not given",
      # d
      "Family member made me feel special",
      # e
      "Had to wear dirty clothes",
      # f
      "Felt loved",
      # g
      "Thought parents wished I wasn't born",
      # h
      "Family hit hard, I had to see doctor",
      # i
      "Family hit hard, bruises or marks",
      # k
      "Punished with belt or hard object",
      # l
      "Family looked out for each other",
      # m
      "Family said hurtful things to me",
      # n
      "Believe I was physically abused",
      # o
      "Hit so bad, people noticed",
      # q
      "Felt family member hated me",
      # r
      "Family felt close to each other",
      # s
      "Someone tried to touch me sexually",
      # t
      "Other threaten harm if no sexual acts",
      # u
      "Other tried to force do/watch sexual",
      # w
      "Someone molested me",
      # x
      "Believe I was emotionally abused",
      # y
      "I was taken to doctor if needed" # z
    )
    
    
    ct_domains <-
      c(
        "Emotional Abuse",
        "Emotional Neglect",
        "Physical Abuse",
        "Physical Neglect",
        "Sexual Abuse"
      )
    
    variable_names <-
      if (input == "ct_items") {
        c(demographics, ct_items)
      } else if (input == "ct_domains") {
        c(demographics, ct_domains)
      } else{
        c(demographics)
      }
    
    
    permutation_plot <- permutation_importance$results %>%
      mutate(across(matches("importance"), ~ . * -1)) %>%
      mutate(
        domain = case_when(
          str_detect(feature, "B4Q9[C|H|N|R|Y]|B4QCT_EA") ~ "Emotional Abuse",
          str_detect(feature, "B4Q9[I|K|L|O|Q]|B4QCT_PA") ~ "Physical Abuse",
          str_detect(feature, "B4Q9[T|U|W|X]|B4Q9AA|B4QCT_SA") ~ "Sexual Abuse",
          str_detect(feature, "B4Q9[E|G|M|S]|B4Q9BB|B4QCT_EN") ~ "Emotional Neglect",
          str_detect(feature, "B4Q9[A$|B$|D|F|Z]|B4QCT_PN") ~ "Physical Neglect",
          TRUE ~ "Demographics"
        )
      ) %>% arrange(feature) %>% mutate(name = variable_names) %>%
      ggplot(., aes(
        color = factor(
          domain,
          levels = c(
            "Emotional Abuse",
            "Physical Abuse",
            "Sexual Abuse",
            "Emotional Neglect",
            "Physical Neglect",
            "Demographics"
          )
        ),
        x = importance,
        y = reorder(name, importance)
      )) +
      geom_linerange(
        mapping = aes(xmin = importance.05, xmax = importance.95),
        show.legend = FALSE
      ) +
      geom_vline(xintercept = 0, linetype = "dashed") +
      geom_point(stat = "identity", size = 3) +
      xlab("\n Decrease in R²") +
      ylab("Feature\n") +
      scale_color_manual(
        values = c(
          "#4DBBD5FF",
          "#8491B4FF",
          "#3C5488FF",
          "#E64B35FF",
          "#F39B7FFF",
          "#B09C85FF"
        )
      ) +
      theme_classic() +
      theme(
        axis.title = element_text(
          size = 10,
          color = "black",
          face = "bold"
        ),
        axis.text = element_text(size = 9.5, color = "black"),
        legend.text =  element_text(size = 9.5, color = "black"),
        legend.title =  element_blank(),
        legend.position = c(0.75, 0.2)
      )
    
    return(permutation_plot)
    
  }



ale <- function(trained_model, input) {
  input_string <-
    case_when(
      input == "ct_items" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|RA4Q9[A-I]|RA4Q9[K-O]|RA4Q9[Q-U]|RA4Q9[W-Z]|^RA4QCESDDA$",
      input == "ct_domains" ~ "MRID|SAMPLMAJ|RA1PRSEX|^RA4ZAGE$|^RA4QCT_[PA|EA|SA|EN|PN]|^RA4QCESDDA$",
      input == "ea_items" ~ "MRID|SAMPLMAJ|^RA4ZAGE$|RA1PRSEX|RA4Q9[C|H|N|R|Y]|^RA4QCESDDA$",
      TRUE ~  "MRID|SAMPLMAJ|^RA4ZAGE$|RA1PRSEX|^RA4QCESDDA$"
    )
  
  pred_data_test <- da36901.0001 %>%
    dplyr::select(.,
                  matches(input_string)) %>%
    dplyr::filter(RA4ZAGE >= 35 &
                    RA4ZAGE <= 60) %>% # same age limit
    mutate(outcome = RA4QCESDDA) %>%
    dplyr::select(-c(MRID, SAMPLMAJ, RA4QCESDDA, RA4ZAGE)) %>%
    dplyr::select(sort(names(.))) %>% # to ensure correct naming later on
    mutate(across(matches("RA1PRSEX"), ~ as.factor(toupper(.)))) %>% # for correspondence with training data set
    dplyr::select(sort(names(.))) %>%
    mutate(across(matches("RA4Q9"), ~ ordered(toupper(.)))) %>%  # to ensure correct collapsing of factor levels
    mutate(across(matches("RA4Q9_"), ~ as.numeric(.))) %>%  # to ensure correct collapsing of factor levels
    filter(!is.na(outcome)) %>%
    filter(rowSums(is.na(.)) / (ncol(.) - 1) <= 0.25) %>%
    dplyr::select(outcome, everything()) # to ensure correct ordering of vars
  
  colnames(pred_data_test) <-
    c("outcome", sort(
      names(
        trained_model$model$regr.ranger$model$variable.importance
      )
    )) # to ensure correct application of the model
  
  
  task_test <- TaskRegr$new("midus",
                            pred_data_test,
                            target = "outcome")
  
  # preprocessing of test data with parameters learned on train data
  preproc_graph_test <-
    trained_model$graph$pipeops$collapsefactors %>>%
    trained_model$graph$pipeops$fixfactors  %>>%
    trained_model$graph$pipeops$fact_to_num %>>%
    trained_model$graph$pipeops$impute_VIM_kNN_B  %>>%
    trained_model$graph$pipeops$num_to_ordered_fact %>>%
    trained_model$graph$pipeops$num_to_unordered_fact
  
  preproc_graph_test$train(task_test)
  
  preproc_graph_test <-
    preproc_graph_test$predict(task_test)
  
  preproc_data_test <-
    as_tibble(preproc_graph_test$num_to_unordered_fact$data())
  
  pfun <-
    function(object, newdata)
      predict(object, data = newdata)$predictions
  
  predictor_dep_test <-
    Predictor$new(
      trained_model$model$regr.ranger$model,
      data = preproc_data_test %>% dplyr::select(-outcome),
      y = preproc_data_test %>% dplyr::select(outcome),
      predict.fun = pfun
    )
  
  
  ale_list <- list()
  for (j in seq_along(task_test$feature_names)) {
    ale_list[[j]] <-
      FeatureEffects$new(predictor_dep_test,
                         task_test$feature_names[j],
                         method = "ale")
  }
  
  return(ale_list)
  
}


# some plotting settings
limits <- c(-0.5, 1.5)
my_theme <- theme_classic() +
  theme(
    axis.title = element_text(size = 10, color = "black"),
    axis.text = element_text(size = 9, color = "black"),
    legend.text =  element_text(size = 9, color = "black"),
    legend.title =  element_blank(),
    legend.position = c(0.75, 0.2)
  )


decompose_r2 <- function(m_conf, m_pred, m_conf_pred) {
  r2_conf <- summary(m_conf)$r.squared
  r2_pred <- summary(m_pred)$r.squared
  r2_conf_pred <- summary(m_conf_pred)$r.squared
  
  conf_unexplained <- 1 - r2_conf
  pred_unexplained <- 1 - r2_pred
  
  delta_pred <- r2_conf_pred - r2_conf
  delta_conf <- r2_conf_pred - r2_pred
  
  partial_pred <- delta_pred / conf_unexplained
  partial_conf <- delta_conf / pred_unexplained
  
  shared = r2_conf_pred - delta_conf - delta_pred
  
  res <- c(
    'confounds' = r2_conf,
    'predictions' = r2_pred,
    'confounds+predictions' = r2_conf_pred,
    'delta confounds' = delta_conf,
    'delta predictions' = delta_pred,
    'partial confounds' = partial_conf,
    'partial predicitons' = partial_pred,
    'shared' = shared
  )
  res <- as.data.frame(res)
  res$r2_type <- rownames(res)
  return(res)
}
