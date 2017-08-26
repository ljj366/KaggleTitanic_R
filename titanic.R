## set working directory
setwd('E:/study/ML/Practice/Titanic')

library(plyr);library(ggplot2)

###################### import data ######################################
train.raw <- read.csv('train.csv',na.strings = c("NA",""),stringsAsFactors=FALSE)
test.raw <- read.csv('test.csv',na.strings = c("NA",""),stringsAsFactors=FALSE)
rlt <- read.csv('gender_submission.csv',na.strings = c("NA",""),stringsAsFactors=FALSE)

# combine train and test
test.raw$Survived <- NA
Data <- rbind(train.raw,test.raw)

########## check cor ##############
# must be numeric, and no NAs 
Data$Sex <- as.numeric(as.factor(Data$Sex))
cor( na.omit(Data)[,c('Survived','Pclass','Age','Sex','SibSp','Parch','Fare')] )

#################### missing values #################################
colSums(is.na(Data))#calculate nb of missing values for each feature

## Approximate missing ages
## use the median of the age to approximate for each category implied by the titles in the names

Data$Title <- str_match(Data$Name,"[a-zA-Z]+\\.") #extract titles
missing.age <- table( Data[is.na(Data$Age), 'Title'] )#list freq of titles whose ages are missing
missing.age <- names(missing.age[missing.age>0]) #get names of titles which are missing

for (i in missing.age){
  Data[is.na(Data$Age) & Data$Title==i,]$Age <- median( Data[!is.na(Data$Age) & Data$Title==i,]$Age )
}

sum(is.na(Data$Age))

## Approximate Embarked ##
## No matter which class, most people go to "S", so use "S" for missing embarked
table(Data$Embarked, Data$Pclass) / matrix(rep(colSums( table(Data$Embarked, Data$Pclass) ),3),nrow=3,byrow=TRUE)
Data[is.na(Data$Embarked),]$Embarked <- names(which.max( table(Data$Embarked) ))

# Approximate Fare #
ggplot(Data, aes(x=factor(Pclass),y=Fare)) + geom_boxplot()
# fare almost determines Pclass, so use average of the Pclass to appox fare
Data[is.na(Data$Fare),]$Fare <- median( table(Data[Data$Pclass == Data[is.na(Data$Fare),'Pclass'],]$Fare) )

colSums(is.na(Data))

#### Create new features 
# 1st character on cabin seems to be deck
Data$Deck <- as.factor(substr(Data$Cabin, 1,1)) #too many missing values, drop it

# family size
Data$Fnb <- Data$SibSp + Data$Parch +1
# plot Fnb vs survived nb
ggplot(Data=Data[!is.na(Data$Survived),], aes(x=Fnb,fill=factor(Survived))) +
  geom_bar(stat='count',position='dodge') + labs(x='Family number') 

# seems families with size bw 2 and 4 are more likely to survive, categorize it
Data$Fsize <- 'Singleton'
Data[Data$Fnb>1 & Data$Fnb<=4,]$Fsize <- 'Small'
Data[Data$Fnb>4,]$Fsize <- 'Large'
table(Data$Fsize)
Data$Fnb <- NULL

# age category
Data$Youth <- 'Adult'
Data[Data$Age<=18,]$Youth <- 'Child'

# Title, we have created title, now combine some
Data$Title <- str_match(Data$Name,"[a-zA-Z]+\\.") #extract titles

Data[Data$Title %in% c('Mlle.','Ms.','Mme.','Miss.','Mrs.','Mr.'),]$Title <- 'Ordinary'
table(Data$Title)
Data[!Data$Title %in% c('Master.','Ordinary'), ]$Title <- 'Other'

#Data[Data$Title %in% c('Mlle.','Ms.','Mme.'),]$Title <- 'Mrs.'
#table(Data$Title)
#Data[!Data$Title %in% c('Master.','Mrs.','Miss.','Mr.'), ]$Title <- 'Other'

Data$Youth <- as.numeric(as.factor(Data$Youth))
Data$Title <- as.numeric(as.factor(Data$Title))
cor( na.omit(Data)[,c('Survived','Pclass','Age','Sex','SibSp','Parch','Fare','Youth','Title')] )

# cahnge categorical variables to factor
factor_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Fsize','Youth','Survived')
Data[factor_vars] <- lapply(Data[factor_vars], function(x) as.factor(x))


##################### Prediction ##############################################
train <- Data[!is.na(Data$Survived) ,]
test <- Data[is.na(Data$Survived),]

### Use all training data ###

## A. random forest
# selecting variables: Fare and Pclass are highly cor. Considering that the captain may give priority to first class, but not distinguish bw fare, so use Pclass is sufficient
model <- randomForest( Survived ~ Title + Sex + Youth + Pclass +Fsize, data=train )

pred <- data.frame( PassengerId=as.integer(test$PassengerId), pred.survived=as.integer(predict(model, test))-1 )
pred <- merge(pred, rlt, by='PassengerId')
mean(pred$pred.survived==pred$Survived)

# check the importance of features
importance <- importance(model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))
varImportance %>% mutate(Rank = paste0('#',rank(desc(Importance))))


## B. logistic 
model <- glm(Survived ~ Title + Sex + Youth + Pclass +Fsize,family=binomial(link='logit'),data=train)
summary(model)
pred <- data.frame( PassengerId=as.integer(test$PassengerId), pred.survived=ifelse(predict(model, test,type='response')>0.5,1,0) )
pred <- merge(pred, rlt, by='PassengerId')
mean(pred$pred.survived==pred$Survived)

###  Cross-validation  ###
library(caret)

## test options #
# use the standard 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3)
metric <- "Accuracy"
formu <- Survived ~ Title + Sex + Age + Pclass + SibSp + Parch + Fare 

# logit 
set.seed(0816)
fit.lg <- train(formu, data=train, method="glm", metric=metric, trControl=trainControl)

# LDA
set.seed(0816)
fit.lda <- train(formu, data=train, method="lda", metric=metric, trControl=trainControl)

# SVM
set.seed(0816)
grid <- expand.grid(.sigma=c(0.025, 0.05, 0.1, 0.15), .C=seq(1, 10, by=1))
fit.svm <- train(formu, data=train, method="svmRadial", metric=metric, tuneGrid=grid,trControl=trainControl)

# random forest
set.seed(0816)
fit.rf <- train(formu, data=train, method="rf", metric=metric, trControl=trainControl)

# KNN
set.seed(0816)
fit.knn <- train(formu, data=train, method="knn", metric=metric, trControl=trainControl)

# compare results
summary( resamples(list(LG=fit.lg, LDA=fit.lda, KNN=fit.knn,SVM=fit.svm, RF=fit.rf)) )

pred <- data.frame( rlt, lg=predict(fit.lg, test), lda=predict(fit.lda,test),knn=predict(fit.knn,test),
                    svm=predict(fit.svm,test),rf=predict(fit.rf,test) )
apply(pred[,-(1:2)], 2, function(x) mean(x==pred[,2]))

# both training and prediction results suggest rf, so report rf results..
write.csv( data.frame(PassengerId = pred$PassengerId,Survived=pred[,'rf']), file = "Titanic_pred.csv", row.names = FALSE)
