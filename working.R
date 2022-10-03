rm(list=ls())
gc()

library(dplyr)
library(pROC)
library(ROCR)
library(glmnet)
library(purrr)
library(ggplot2)
set.seed(123)

###### read and process data #####
dataset_index <- c("australian.dat", #"german.data", 
                   "kaggle-cs-training.csv",
                   "AER_credit_card_data.csv")
tmp_dataset <- dataset_index[3]

if(tmp_dataset == "kaggle-cs-training.csv"){
  df <- read.csv(paste0("data/",tmp_dataset))
  df$X <- NULL
  df <- df[, c(2:ncol(df), 1)]
} else if(tmp_dataset == "AER_credit_card_data.csv"){
  df <- read.csv(paste0("data/",tmp_dataset))
  df <- df[, c(2:ncol(df), 1)]
} else{
  df <- read.table(paste0("data/",tmp_dataset), 
                   header=FALSE)
}

df[is.na(df)] <- 0

colnames(df)[length(colnames(df))] <- "target"

df[] <- lapply(df,function(x) as.numeric(as.factor(x)))

if(tmp_dataset %in% c("australian.dat","german.data","kaggle-cs-training.csv")){
  df$target <- ifelse(df$target==1,0,1)
} else if(tmp_dataset == "AER_credit_card_data.csv"){
  df$target <- ifelse(df$target==2,0,1)
}

# table(df$target)
# dim(df)




###### train test split #######
smp_size <- floor(0.98 * nrow(df))
dev_ind <- sample(seq_len(nrow(df)), size = smp_size)
df_train <- df[dev_ind, ]
df_test <- df[-dev_ind, ]

mean(df$target)
mean(df_train$target)
mean(df_test$target)

# assume first five coefficients to be non-positive
# last five receive no penalty

##### MODELLING ######
auc_lasso <- NULL
auc_cpr <- NULL

for(i in 1:20){
  print(paste("Current iteration:",i))
  #i=1
  ######## LASSO ######
  # uses misclassification error as the criterion for 10-fold cross-validation
  set.seed(i)
  cv_lasso <- cv.glmnet(as.matrix(df_train[, !(colnames(df_train) %in% "target")]), 
                        df_train$target,
                        family = "binomial", 
                        type.measure = "class",
                        nfolds = 5,
                        nlambda = 50,
                        alpha=1,
                        penalty.factor = c(rep(1,ncol(df_train)-1-5), rep(0.000001,5)),
                        upper.limits=c(rep(0,5),rep(Inf, ncol(df_train)-1-5))
  )
  # plot(cv_lasso)
  # model_lasso <- glmnet(df_train[, !(colnames(df_train) %in% "target")], 
  #                       df_train$target, alpha = 1, family = "binomial",
  #                 lambda = cv_lasso$lambda.min)
  # Make predictions on the test data
  probabilities1 <- predict(cv_lasso, 
                           as.matrix(df_test[, !(colnames(df_test) %in% "target")]),
                           s = "lambda.min",
                           type = "response")
  pred1 <- prediction(probabilities1, df_test$target)
  tmp_auc_lasso <- as.numeric(performance(pred1, "auc")@y.values)
  auc_lasso <- c(auc_lasso, tmp_auc_lasso)
  
  ###### CPR-LR ######
  get_correlation <- function(col){
    return(cor(df_train$target,df_train[, col]))
  }
  
  correls <- purrr::map(colnames(df_train)[-ncol(df_train)], get_correlation) %>% unlist()
  
  cv_cpr <- cv.glmnet(as.matrix(df_train[, !(colnames(df_train) %in% "target")]), 
                      df_train$target,
                      family = "binomial", 
                      type.measure = "class",
                      nfolds = 5,
                      nlambda = 100,
                      alpha=0.8,
                      penalty.factor = c(correls[1:(ncol(df_train)-1-5)], rep(0.000001,5)),
                      upper.limits=c(rep(0,5),rep(Inf, ncol(df_train)-1-5))
  )
  
  probabilities2 <- predict(cv_cpr, 
                           as.matrix(df_test[, !(colnames(df_test) %in% "target")]),
                           s = "lambda.min",
                           type = "response")
  pred2 <- prediction(probabilities2, df_test$target)
  tmp_auc_cpr <- as.numeric(performance(pred2, "auc")@y.values)
  auc_cpr <- c(auc_cpr, tmp_auc_cpr)
}

write.csv(cbind(auc_lasso, auc_cpr),"tmp.csv")


##### PLOTTING #####
results <- read.csv("credit risk modelling experiment result - Sheet1.csv")
colnames(results) <- c("Dataset", "LASSO", "CPR-LR")

library(data.table)
long <- melt(setDT(results), id.vars = c("Dataset"), variable.name = "Model")
colnames(long)[3] <- "ROC_AUC"

# s1 <- ggplot(long, aes(x = Dataset, y = ROC_AUC, color = Model)) +
#   geom_boxplot(position = position_dodge(width = 0.7)) +
#   geom_point(position = position_jitterdodge(seed = 123))
# # s1

long %>% group_by(Dataset, Model) %>% 
  summarise(Mean = mean(ROC_AUC),
            Median = median(ROC_AUC))

# roc curve
# i=20 for australian credit; 1 for kaggle competition
roc1 <- plot.roc(df_test$target, as.vector(probabilities1), #main="ROC comparison", 
                 percent=FALSE, col= "blue")#, label="LASSO")
roc2 <- lines.roc(df_test$target, as.vector(probabilities2), percent=FALSE, 
                  col="red")#, label="CPR-LR")

legend("topright", c("LASSO", "CPR-LR"), lty=10, 
       col = c("blue", "red"), bty="n", inset=c(0,0.75))

