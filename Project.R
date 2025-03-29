############# Helper Functions #####################
####################################################
installpkg <- function(x){
  if(x %in% rownames(installed.packages())==FALSE) {
    if(x %in% rownames(available.packages())==FALSE) {
      paste(x,"is not a valid package - please check again...")
    } else {
      install.packages(x)           
    }
    
  } else {
    paste(x,"package already installed...")
  }
}

RMSE <- function(y, pred) {
  sqrt(mean( (y - pred )^2))
}

support<- function(x, tr = 10e-6) {
  m<- rep(0, length(x))
  for (i in 1:length(x)) if( abs(x[i])> tr ) m[i]<- i
  m <- m[m>0]
  m
}

############ Import data ###############
setwd("~/Downloads/")
spotify_songs <- read.csv("spotify_songs.csv")
## number of songs with 0s for track popularity seems very disproporionate
## these are NULL values, maybe these are NULL values?
hist(spotify_songs$track_popularity)
## 8% of the songs have 0s as track_popularity
nrow(spotify_songs[spotify_songs$track_popularity < 10,])/ nrow(spotify_songs)
## the 0-10 bracket is still disproportionately large, so we assume they're not NULL values.
hist(spotify_songs[spotify_songs$track_popularity != 0,]$track_popularity)

######### Exploratory Data Analysis ##############
installpkg("ggplot2")
library(ggplot2)
installpkg("dplyr")
library(dplyr)
installpkg("factoextra")
library(factoextra)

ggplot(spotify_songs, aes(x = playlist_genre)) +
  geom_bar(aes(fill = playlist_genre), show.legend = FALSE) +
  labs(title = 'Count of Each Genre', x = 'Genre', y = 'Count') +
  theme_minimal()

############# Popularity by Genre #######################
genre_popularity <- spotify_songs %>%
  group_by(playlist_genre) %>%
  summarise(avg_popularity = mean(track_popularity))

ggplot(spotify_songs, aes(x = reorder(playlist_genre, track_popularity, FUN = median), y = track_popularity, fill = playlist_genre)) +
  geom_boxplot() +
  labs(title = 'Track Popularity by Genre', x = 'Genre', y = 'Track Popularity') +
  theme_minimal()

############# PCA Analysis #######################
features <- spotify_songs[, c("danceability", "energy", "key", "loudness", "mode", 
                        "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms")]

features_scaled <- scale(features)
pca_result <- prcomp(features_scaled, center = TRUE, scale. = TRUE)
summary(pca_result)

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 50)) +
  labs(title = "PCA: Variance Explained by Factors")

print(pca_result$rotation[, 1:4])

########## Tempo vs. Danceability ###

ggplot(spotify_songs, aes(x = tempo, y = danceability)) +
  geom_point(color = "lightblue", alpha = 0.5, size = 3) + 
  geom_smooth(method = "loess", color = "blue", se = FALSE) + 
  labs(title = 'Tempo vs. Danceability', x = 'Tempo (BPM)', y = 'Danceability') +
  theme_minimal()


## we drop columns that we won't use for our prediction models and scale track_popularity to be within 0 and 1
data <- spotify_songs[,!names(spotify_songs) %in% c('track_id', 'track_album_id', 'track_name', 'track_artist', 
                                                    'track_album_name', 'track_album_name', 'playlist_name', 'playlist_id')]
data$track_popularity <- data$track_popularity /100
## no missing values
sum(is.na(data))

## explore correlations between the numerical variables
installpkg("corrplot")
library(corrplot)
matrix <- cor(data[,!names(data) %in% c('track_album_release_date', 'playlist_genre', 'playlist_subgenre')])
corrplot(matrix)

## only include the year information from date as there are a lot of songs with missing month and day data
data['Year'] <- as.numeric(sub("^(\\d{4})-.*|^(\\d{4})$", "\\1\\2", data$track_album_release_date))
data <- data[,!names(data) %in% c('track_album_release_date')]



######################################
############# Modeling ###############
######################################
installpkg("glmnet")
library(glmnet)
installpkg("randomForest")
library(randomForest)
installpkg("tree")
library(tree)
installpkg("partykit")
library(partykit)
installpkg("caret")
library(caret)
installpkg("rpart")
library(rpart)
installpkg("rpart.plot")
library(rpart.plot)
installpkg("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
library(keras)
library(tensorflow)
library(reticulate)
use_condaenv("data_science", required=TRUE)

set.seed(17)
### create a vector of fold memberships (random order)
n <- nrow(data)
nfold <- 10
foldid <- rep(1:nfold,each=ceiling(n/nfold))[sample(1:n)]

OOSPerformance <- data.frame(lin.reg=rep(NA,nfold), lasso=rep(NA,nfold), 
                             post.lasso=rep(NA,nfold), rf=rep(NA,nfold),  
                             tree=rep(NA,nfold), nn=rep(NA,nfold))

InSamplePerformance <- data.frame(null=NA, lin.reg=NA, 
                                  lasso=NA, post.lasso=NA, 
                                  rf=NA, tree=NA, nn=NA)

##### Null Model #######
My <- data$track_popularity
prednull <- mean(data$track_popularity)*rep(1,length(data$track_popularity))
InSamplePerformance$null <- RMSE(data$track_popularity, prednull)

##### Linear Regression ######
lin.reg <- glm(track_popularity~., data=data)
pred.lin.reg <- predict(lin.reg, newdata=data, type="response")
InSamplePerformance$lin.reg <- RMSE(data$track_popularity, pred.lin.reg)

###### Lasso #########
Mx <- model.matrix(track_popularity ~ .^2, data=data)[,-1]
lasso <- glmnet(Mx,My)
lassoCV <- cv.glmnet(Mx,My)
par(mar=c(1.5,1.5,2,1.5))
par(mai=c(1.5,1.5,2,1.5))
plot(lassoCV, main="Fitting Graph for CV Lasso \n \n # of non-zero coefficients  ", xlab = expression(paste("log(",lambda,")")))

lassomin  <- glmnet(Mx,My,lambda = lassoCV$lambda.min)
predlassomin <- predict(lassomin, newx=Mx, type="response")
InSamplePerformance$lasso <- RMSE(data$track_popularity, predlassomin)

###### Post-Lasso #######
features.min <- support(lasso$beta[,which.min(lassoCV$cvm)])
length(features.min)
data.min <- data.frame(Mx[,features.min],My)

pl <- glm(My~., data=data.min)
predpostlasso <- predict(pl, newdata=data.min, type="response")
InSamplePerformance$post.lasso <- RMSE(data$track_popularity, predpostlasso)

##### Random Forest ##########

model.rf <- randomForest( track_popularity~., data=data)
round(importance(model.rf), 2)
randomForest::varImpPlot(model.rf, 
                         sort=FALSE, 
                         main="Variable Importance Plot")
pred.rf <- predict(model.rf, newdata=data)
InSamplePerformance$rf <- RMSE(data$track_popularity, pred.rf)

###### TREE ############
model.tree <- rpart(track_popularity~., data=data, cp=0.008)
rpart.plot(model.tree)
pred.tree <- predict(model.tree, newdata=data)
InSamplePerformance$tree <- RMSE(data$track_popularity, pred.tree)

###### Deep Learning #########

x_train <- Mx
max_vals <- apply(x_train, 2, max)
max_vals[max_vals == 0] <- 1
norm_vec <- 1 / max_vals
x_train <- (x_train %*% diag(norm_vec))
apply(x_train, 2, max)
y_train <- data$track_popularity

num.inputs <- ncol(x_train)
num.inputs

model <- keras_model_sequential() %>%
  layer_dense(units=16,activation="relu",input_shape = c(num.inputs)) %>%
  layer_dense(units=16,activation="relu") %>%
  layer_dense(units=16,activation="relu") %>%
  layer_dense(units=1,activation="sigmoid")


summary(model)

model %>% compile(
  loss = 'MeanSquaredError',
  optimizer = optimizer_rmsprop()
  #metrics = c('accuracy')
)
history <- model %>% fit(
  x_train, y_train, 
  epochs = 60, batch_size = 64, 
  validation_split = 0.1
)
results.NN1 <- model %>% evaluate(x_train,y_train)
results.NN1

pred.nn <- model %>% predict(x_train)
InSamplePerformance$nn <- RMSE(data$track_popularity, pred.nn)



###### In Sample Evaluation ########
names(InSamplePerformance) <- c("Null Model", "Linear Regression", "Lasso", "Post-Lasso", "Random Forest", "Tree",  "Deep Learning \n (two hidden layers)")
par(mar = c(10, 5, 4, 2))
barplot(colMeans(InSamplePerformance), col= "blue", las=2,xpd=FALSE , 
        xlab="", ylab = bquote( "In-Sample RMSE Performance"))

###### Cross Validation #######

for(k in 1:nfold){ 
  train <- which(foldid!=k) # train on all but fold `k'
  
  ### CV for the Post Lasso Estimates
  pl <- glm(My~., data=data.min, subset=train)
  predmin <- predict(pl, newdata=data.min[-train,], type="response")
  OOSPerformance$post.lasso[k] <- RMSE(My[-train], predmin)
  
  ### CV for the Lasso estimates  
  lassomin  <- glmnet(Mx[train,],My[train],lambda = lassoCV$lambda.min)
  predlassomin <- predict(lassomin, newx=Mx[-train,], type="response")
  OOSPerformance$lasso[k] <- RMSE(My[-train], predlassomin)
  
  ### CV for Linear Regression model
  model.lreg <- glm(track_popularity~., data=data, subset=train)
  pred.lreg <- predict(model.lreg, newdata=data[-train,], type="response")
  OOSPerformance$lin.reg[k] <- RMSE(My[-train], pred.lreg)
  
  ### model.rf
  model.rf <- randomForest( track_popularity~., data=data, subset=train)
  pred.rf <- predict(model.rf, newdata=data[-train,], type="response")
  OOSPerformance$rf[k] <- RMSE(My[-train], pred.rf)
  
  ### model.tree
  model.tree <- rpart(track_popularity~., data=data[train,], cp=0.008)
  pred.tree <- predict(model.tree, newdata = data[-train,])
  OOSPerformance$tree[k] <- RMSE(My[-train], pred.tree)
  
  ### nn
  history <- model %>% fit(
    x_train[train,], y_train[train], 
    epochs = 60, batch_size = 64, 
    validation_split = 0.1
  )
  pred.nn <- model %>% predict(x_train[-train,])
  OOSPerformance$nn[k] <- RMSE(My[-train], pred.nn)
  
  ###
  print(paste("Iteration",k,"of",nfold,"completed"))
}

######## Out of sample evaluation #########
names(OOSPerformance) <- c("Linear Regression", "Lasso", "Post-Lasso", "Random Forest", "Tree", "Deep Learning \n (two hidden layers)")

par(mar = c(10, 5, 4, 2))
barplot(colMeans(OOSPerformance), col= "blue", las=2,xpd=FALSE , 
        xlab="", ylab = bquote( "Out of Sample RMSE Performance"))


######## Random forest is our chosen model ###########

