#This dataset was imported from Kaggle.com .
#The dataset includes several months of data on daily trending YouTube videos. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count.
#Dataset prediction - Analysing what factors affect how popular a YouTube video will be
#Main features in the dataset are: views,comment_count,likes,dislikes.We like to determine which features are best for predicting the views of a Youtube trending video.We suspect comment_count,likes,dislikes and some combination of the other features could be used to build a predictive model to determine views count of a Youtube trending video.

Data<- read.csv("G:/Machine Learning/Project/INvideos.csv")
names(Data)
dim(Data)
attach(Data)
#preprocessing
Data["comments_disabled"] <- as.integer(Data$comments_disabled)
Data["ratings_disabled"] <- as.integer(Data$ratings_disabled)
Data["video_error_or_removed"] <- as.integer(Data$video_error_or_removed)


Data["trending_date"]=as.integer(as.Date(Data$trending_date[dim(Data)[1]],format="%y.%d.%m") - as.Date(Data$trending_date,format="%y.%d.%m"))
Data$category_id <- factor(Data$category_id)

myvars <- c("trending_date" ,"category_id","likes", "dislikes","comment_count","comments_disabled","ratings_disabled","video_error_or_removed","views")
final <- Data[myvars]
names(final)
summary(final)

#normalize
normalise <- function(x) { + return ( ( x - min(x)) / ( max(x) - min(x))) }
final1<- as.data.frame(lapply(final[,c(3,4,5)], normalise))
final$likes<-final1$likes
final$dislikes<-final1$dislikes
final$comment_count<-final1$comment_count

#shuffle
set.seed(9850)
gp <- runif(nrow(final))
head(gp)
final<- final[order(gp), ]
head(final)



#model with residual plt
fit = lm(views ~ trending_date+category_id+likes+dislikes+comment_count+comments_disabled+ratings_disabled+video_error_or_removed, data=final)
summary(fit)
plot(fit)

fit2=lm(views~likes*likes+dislikes)
plot(fit2)
summary(fit2)

fit1=lm(views ~ trending_date+category_id+likes*likes+dislikes*dislikes+comment_count+video_error_or_removed, data=final)
summary(fit1)

plot(fit1)
#Only taking important factors into consideration for fitting the model
#R square is greater for the one with all the factor which is better than R squared value of model with Likes and Dislikes and which has better R squared value than the polynomial plot using Likes^2
#In the dataset there are some Outliers present which can be seen in the Residual Plot
library(ISLR)
library(boot)
library(MASS)

#train and test
train = sample(1:nrow(final), nrow(final)/2)
final_train = final[train,]
final_test = final[-train,]
fit = lda(views ~ trending_date+likes+comments_disabled+ratings_disabled+video_error_or_removed, data=final_train)
pred=predict(fit, final_test)
pred_class=pred["class"]
dim(table(pred["class"], final_test["views"]))
#mean(pred_class == final_test$views)
dim(pred)
summary(pred)
class(Data$category_id)
summary(fit)
#bootstrap

alpha.fn=function(data,index){
  X=data$X[index]
  Y=data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
}
alpha.fn(Portfolio,1:100)
set.seed(1)
alpha.fn(Portfolio,sample(100,100,replace=T))
boot(Portfolio,alpha.fn,R=1000)

#Forward and Backward Stepwise Selection
library(leaps)
regfit.fwd=regsubsets(Data$views~.,data=final,nvmax=19,method="forward")
summary(regfit.fwd)
regfit.bwd=regsubsets(Data$views~.,data=final,nvmax=19,method="backward")
summary(regfit.bwd)
coef(regfit.fwd,7)
coef(regfit.bwd,7)


#CV
set.seed(17)
cv.error.10=rep(0,10)
for (i in 1:10){
  glm.fit=glm(views~.,data=final)
  cv.error.10[i]=cv.glm(final,glm.fit,K=10)$delta[1]
}
cv.error.10