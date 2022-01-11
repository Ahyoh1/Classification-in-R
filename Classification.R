setwd("/Users/victoryogbebor/Documents/Documents - Victory’s MacBook Air/Classification-in-R")

default <- read.csv("default of credit card clients.csv")


library(Boruta)
library(mlbench)
library(caret)
library(ROSE)
library(party)
library(gbm)
library(e1071)
library(class)
library(highcharter)
library(randomForest)

names(default)
head(default)
tail(default)
summary(default)
str(default)
dim(default) 

# Feature Selection
set.seed(555)
default_br <- Boruta(default.payment.next.month ~ ., data = default, 
                     doTrace = 2, maxRuns = 250)
print(default_br)
plot(default_br, las = 2, cex.axis = 0.7, ylim=c(0,45))

#Remove education
default_ <- default[,-c(4)]
#Convert columns to factors
cols <- c("default.payment.next.month", "SEX", "MARRIAGE")
default_[cols] <- lapply(default_[cols], factor)


# Data Partition
nrows <- NROW(default_)
set.seed(128)
ind <- sample( 1:nrows, 0.7 * nrows)
train <- default_[ind,]
test <- default_[-ind,]



#Check for class imbalance in the training set
table(train["default.payment.next.month"])

barplot(prop.table(table(default_$default.payment.next.month)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

default_und <- ovun.sample(default.payment.next.month~., data=train, 
                           method = "under", N = 9100)$data
table(default_und$default.payment.next.month)

barplot(prop.table(table(default_und$default.payment.next.month)),
        col = rainbow(2),
        ylim = c(0, 1),
        main = "Class Distribution")

train <- default_und
#Check for class imbalance in the new training set
table(train["default.payment.next.month"])

#Decision tree
learn_df <- ctree(default.payment.next.month~., data=train, 
                  controls=ctree_control(maxdepth=7))
pre_df   <- predict(learn_df, test[,-24])
pre_df1 <- as.factor(pre_df)
cm_ct    <- confusionMatrix(pre_df, test$default.payment.next.month)
cm_ct


#classification accuracy
df_acc <- sum(diag(cm_ct))/sum(cm_ct)
print(paste("The decision tree model is", (round(df_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(df_acc, digits = 4)))*100, "%", "error rate"))

#Logistic regression
learn_lr <- glm(default.payment.next.month~., data=train, family = "binomial")
summary(learn_lr)
learn_lr2 <- glm(default.payment.next.month ~ LIMIT_BAL + SEX + PAY_0 
                 + BILL_AMT1 + PAY_AMT2 + PAY_AMT1 + PAY_AMT5,
                 data=train, family = "binomial")
pre_lr <- predict(learn_lr2, test[,-24], type = 'response')
pre_lr1 <- ifelse(pre_lr>0.5, 1, 0)
pre_l <- as.factor(pre_lr1)
cm_lr    <- confusionMatrix(pre_l, test$default.payment.next.month)
cm_lr
#classification accuracy
lr_acc <- sum(diag(cm_lr))/sum(cm_lr)
print(paste("The logistic regression model is", (round(lr_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(lr_acc, digits = 4)))*100, "%", "error rate"))


#RandomForest
learn_rf <- randomForest(default.payment.next.month~., 
                         data=train, ntree=500, proximity=T, importance=T)
pre_rf   <- predict(learn_rf, test[,-24])
cm_rf    <- confusionMatrix(pre_rf, test$default.payment.next.month)
cm_rf
cc_rf <- table(pre_rf, test$default.payment.next.month)
#classification accuracy
rf_acc <- sum(diag(cc_rf))/sum(cc_rf)
print(paste("The random forest model is", (round(rf_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(rf_acc, digits = 4)))*100, "%", "error rate"))

# Knn
acc_test <- numeric() 

for(i in 1:30){
  predict <- knn(train=train[,-24], test=test[,-24], cl=train[,24], k=i, prob=T)
  acc_test <- c(acc_test,mean(predict==test[,24]))
}

acc <- data.frame(k= seq(1,30), cnt = acc_test)

opt_k <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of k is", opt_k$k, "(accuracy :", opt_k$cnt,") in KNN")


hchart(acc, 'line', hcaes(k, cnt)) %>%
  hc_title(text = "Accuracy With Varying K (KNN)") %>%
  hc_subtitle(text = sub) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_xAxis(title = list(text = "Number of Neighbors(k)")) %>%
  hc_yAxis(title = list(text = "Accuracy"))

#Apply optimal K to show best predict performance in KNN
pre_knn <- knn(train = train[,-24], test = test[,-24], cl = train[,24], k=opt_k$k, prob=T)
cm_knn  <- confusionMatrix(pre_knn, test$default.payment.next.month)
cm_knn
kn_rf <- table(pre_knn, test$default.payment.next.month)
#classification accuracy
kn_acc <- sum(diag(kn_rf))/sum(kn_rf)

print(paste("The KNN model is", (round(kn_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(kn_acc, digits = 4)))*100, "%", "error rate"))

#Naive Bayes
acc_test <- numeric()
accuracy1 <- NULL; accuracy2 <- NULL

for(i in 1:100){
  learn_imp_nb <- naiveBayes(train[,-24], train$default.payment.next.month, laplace=i)    
  p_nb <- predict(learn_imp_nb, test[,-24]) 
  accuracy1 <- confusionMatrix(p_nb, test$default.payment.next.month)
  accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(l= seq(1,100), cnt = accuracy2)

opt_l <- subset(acc, cnt==max(cnt))[1,]
sub <- paste("Optimal number of laplace is", opt_l$l, "(accuracy :", opt_l$cnt,") in naiveBayes")

hchart(acc, 'line', hcaes(l, cnt)) %>%
  hc_title(text = "Accuracy With Varying Laplace (naiveBayes)") %>%
  hc_subtitle(text = sub) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_xAxis(title = list(text = "Number of Laplace")) %>%
  hc_yAxis(title = list(text = "Accuracy"))

learn_nb <- naiveBayes(train[,-24], train$default.payment.next.month, laplace = 62)
pre_nb <- predict(learn_nb, test[,-24])
cm_nb <- confusionMatrix(pre_nb, test$default.payment.next.month)        
cm_nb
nb_rf <- table(pre_nb, test$default.payment.next.month)
#classification accuracy
nb_acc <- sum(diag(nb_rf))/sum(nb_rf)
print(paste("The Naive Bayes model is", (round(nb_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(nb_acc, digits = 4)))*100, "%", "error rate"))

#Gradient Boost
test_gbm <- gbm(default.payment.next.month~., data=train, distribution="gaussian",n.trees = 10000,
                shrinkage = 0.01, interaction.depth = 4, bag.fraction=0.5, train.fraction=0.5,n.minobsinnode=10,cv.folds=3,keep.data=TRUE,verbose=FALSE,n.cores=1)
best.iter <- gbm.perf(test_gbm, method="cv",plot.it=FALSE)
fitControl = trainControl(method="cv", number=5, returnResamp="all")
learn_gbm = train(default.payment.next.month~., data=train, method="gbm", distribution="bernoulli", trControl=fitControl, verbose=F, tuneGrid=data.frame(.n.trees=best.iter, .shrinkage=0.01, .interaction.depth=1, .n.minobsinnode=1))
pre_gbm <- predict(learn_gbm, test[,-24])
cm_gbm <- confusionMatrix(pre_gbm, test$default.payment.next.month)
cm_gbm
gb_rf <- table(pre_gbm, test$default.payment.next.month)
#classification accuracy
gb_acc <- sum(diag(gb_rf))/sum(gb_rf)

print(paste("The Gradient boost model is", (round(gb_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(gb_acc, digits = 4)))*100, "%", "error rate"))

#SVM
learn_svm <- svm(default.payment.next.month~., data=train)
pre_svm <- predict(learn_svm, test[,-24])
cm_svm <- confusionMatrix(pre_svm, test$default.payment.next.month)
cm_svm
svm_rf <- table(pre_svm, test$default.payment.next.month)
#classification accuracy
svm_acc <- sum(diag(svm_rf))/sum(svm_rf)

print(paste("The Support Vector model is", (round(svm_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(svm_acc, digits = 4)))*100, "%", "error rate"))

#Tuned SVM
gamma <- seq(0,0.1,0.005)
cost <- 2^(0:5)
parms <- expand.grid(cost=cost, gamma=gamma)    

acc_test <- numeric()
accuracy1 <- NULL; accuracy2 <- NULL

for(i in 1:NROW(parms)){        
  learn_svm <- svm(default.payment.next.month~., data=train, gamma=parms$gamma[i], cost=parms$cost[i])
  pre_svm <- predict(learn_svm, test[,-24])
  accuracy1 <- confusionMatrix(pre_svm, test$default.payment.next.month)
  accuracy2[i] <- accuracy1$overall[1]
}

acc <- data.frame(p= seq(1,NROW(parms)), cnt = accuracy2)

opt_p <- subset(acc, cnt==max(cnt))[1,]
#sub <- paste("Optimal number of parameter is", opt_p$p, "(accuracy :", opt_p$cnt,") in SVM")

hchart(acc, 'line', hcaes(p, cnt)) %>%
  hc_title(text = "Accuracy With Varying Parameters (SVM)") %>%
  hc_subtitle(text = sub) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_xAxis(title = list(text = "Number of Parameters")) %>%
  hc_yAxis(title = list(text = "Accuracy"))

learn_imp_svm <- svm(default.payment.next.month~., data=train, cost=parms$cost[opt_p$p], gamma=parms$gamma[opt_p$p])
pre_imp_svm <- predict(learn_imp_svm, test[,-24])
cm_imp_svm <- confusionMatrix(pre_imp_svm, test$default.payment.next.month)
cm_imp_svm
imp_svm_rf <- table(pre_imp_svm, test$default.payment.next.month)
#classification accuracy
imp_svm_acc <- sum(diag(imp_svm_rf))/sum(imp_svm_rf)

print(paste("The Tuned Support Vector model is", (round(imp_svm_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(imp_svm_acc, digits = 4)))*100, "%", "error rate"))



#Visualize to compare the accuracy of all methods
col <- c("#ed3b3b", "#0099ff")
par(mfrow=c(3,3))
fourfoldplot(cm_ct$table, color = col, conf.level = 0, margin = 1, main=paste("D.Tree (",round(cm_ct$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_lr$table, color = col, conf.level = 0, margin = 1, main=paste("Log Reg (",round(cm_lr$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_nb$table, color = col, conf.level = 0, margin = 1, main=paste("NaiveBayes (",round(cm_nb$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_knn$table, color = col, conf.level = 0, margin = 1, main=paste("Tune KNN (",round(cm_knn$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_rf$table, color = col, conf.level = 0, margin = 1, main=paste("RandomForest (",round(cm_rf$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_gbm$table, color = col, conf.level = 0, margin = 1, main=paste("GBM (",round(cm_gbm$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_svm$table, color = col, conf.level = 0, margin = 1, main=paste("SVM (",round(cm_svm$overall[1]*100),"%)",sep=""))
fourfoldplot(cm_imp_svm$table, color = col, conf.level = 0, margin = 1, main=paste("Tune SVM (",round(cm_imp_svm$overall[1]*100),"%)",sep=""))

#Accuracy
accuracy <- c((paste("The decision tree model is", (round(df_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(df_acc, digits = 4)))*100, "%", "error rate")),
(paste("The logistic regression model is", (round(lr_acc, digits = 4))*100, "%", "accurate and has"
                  , (1-(round(lr_acc, digits = 4)))*100, "%", "error rate")),
(paste("The random forest model is", (round(rf_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(rf_acc, digits = 4)))*100, "%", "error rate")),

(paste("The KNN model is", (round(kn_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(kn_acc, digits = 4)))*100, "%", "error rate")),

(paste("The Naive Bayes model is", (round(nb_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(nb_acc, digits = 4)))*100, "%", "error rate")),

(paste("The Gradient boost model is", (round(gb_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(gb_acc, digits = 4)))*100, "%", "error rate")),

(paste("The Support Vector model is", (round(svm_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(svm_acc, digits = 4)))*100, "%", "error rate")),

(paste("The Tuned Support Vector model is", (round(imp_svm_acc, digits = 4))*100, "%", "accurate and has"
            , (1-(round(imp_svm_acc, digits = 4)))*100, "%", "error rate")))

print(accuracy)
