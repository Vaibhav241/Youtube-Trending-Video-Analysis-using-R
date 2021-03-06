#Kirti Bhagat(17BLC1046)
#Vaibhav Nagpal(17BLC1082)
#This dataset was imported from Kaggle.com .
#The dataset includes several months of data on daily trending YouTube videos. Data includes the video title, channel title, publish time, tags, views, likes and dislikes, description, and comment count.
#Dataset prediction - Analysing what factors affect how popular a YouTube video will be
#Main features in the dataset are: views,comment_count,likes,dislikes.We like to determine which features are best for predicting the views of a Youtube trending video.We suspect comment_count,likes,dislikes and some combination of the other features could be used to build a predictive model to determine views count of a Youtube trending video.
library(ISLR)
library(tree)
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

myvars <- c("trending_date" ,"likes", "dislikes","comment_count","comments_disabled","ratings_disabled","video_error_or_removed","views")
final <- Data[myvars]
names(final)
summary(final)

#normalize
normalise <- function(x) { + return ( ( x - min(x)) / ( max(x) - min(x))) }
final1<- as.data.frame(lapply(final[,c(3,4,5)], normalise))
#shuffle
set.seed(9850)
gp <- runif(nrow(final))
head(gp)
final<- final[order(gp), ]
head(final)
train = sample(1:nrow(final), nrow(final)/2)
final_train = final[train,]
final_test = final[-train,]

library(e1071)
svmfit <- svm(final_train$views ~likes+dislikes+comment_count+comments_disabled+video_error_or_removed, data=final_train,kernel='linear',cost=5, scale=FALSE)
summary(svmfit)
print(svmfit)
plot(svmfit,final_train,fill=TRUE)

pred <- predict(svmfit,final_test)
pred
