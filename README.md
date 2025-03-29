# Music Recommendation System

## Business Understanding

  Music platforms continuously seek strategic ways to help users discover songs they enjoy to increase
user engagement on their platform. As the music industry has become increasingly digitalized,
platforms such as Spotify and Apple Music have found success through personalized music recommendations. By providing better suggestions, these platforms drive higher engagement, longer listening sessions, and potentially better customer retention.
 
  Our goal is to improve the platform’s recommendation system by focusing on predicting the
popularity of a song based on its characteristics. Unlike traditional recommendation systems that
rely heavily on user interaction data, we will focus on the song’s audio features (such as tempo,
energy, danceability, and acousticness) and metadata (such as genre and release year).
 
  We hypothesize that songs with specific patterns in their features are more likely to achieve higher
popularity. For example, songs with higher energy and danceability levels may be more popular in
certain genres, while acoustic songs may perform better in others. By analyzing these patterns, we
can make more accurate recommendations, which in turn increases user satisfaction and engagement
on the platform.

## Data Understanding and Exploratory Data Analysis

### Data Set

The data set consists of various tracks from the Spotify platform. Each entry includes a track
name, artist, track popularity, and more. *Table 1* displays an overview of the dataset. In addition,
characteristics such as danceability, energy, liveness, and more are recorded as scores ranging from
0-1.

<p align="center">
<img width="700" alt="Screenshot 2025-03-28 at 8 58 46 PM" src="https://github.com/user-attachments/assets/785afff8-5e5c-47e7-96ab-42ef0be63b16" />

### Correlation Analysis

*Figure 1* displays the correlations between the variables included in the predictive models. We
don’t see any variables that are highly correlated with track popularity. However, we notice that
accousticness and energy, loudness and energy, and valence and danceability are highly correlated.

<p align="center">
<img width="520" alt="Screenshot 2025-03-28 at 8 59 27 PM" src="https://github.com/user-attachments/assets/90ca338a-8ed4-4fff-bb31-d91e978d97b9" />

### Principal Component Analysis

*Figure 2* shows how the top ten principal components explain the total variance of the data, specifically the percentage of variance explained by each component. The first principal component
explains 17.9% of the total variance, which is significantly higher than the other principal components, indicating that it captures the most dominant features of variation in the data. We also see a gradual decline in the variance captured for each subsequent PCA. 

<p align="center">
<img width="638" alt="Screenshot 2025-03-28 at 9 00 07 PM" src="https://github.com/user-attachments/assets/21b67eb0-a269-45bd-8380-5e32247741bc" />

*Table 2* displays the first four PCA loadings. We can loosely interpret them as the following: PC1 highlights loudness and energy,
with negative acousticness, indicating that energetic songs score higher, while acoustic songs score
lower. PC2 focuses on danceability, valence, and speechiness, where higher scores correspond to
more danceable, positive, and speech-heavy songs. PC3 contrasts mode and key, with higher scores
representing major key songs and lower scores representing minor key songs. PC4 relates to duration, speechiness, and instrumentalness, with higher scores indicating longer, more speech-driven,
and instrumental songs.

<p align="center">
<img width="574" alt="Screenshot 2025-03-28 at 9 00 34 PM" src="https://github.com/user-attachments/assets/17b2d764-014b-43c4-ba59-4ec5f61b2233" />

### Genre & Track Popularity

*Figure 3* shows the relationship between genre and track popularity to identify any patterns that
could inform our recommendation model. As shown in the box plot, genres like Pop and Latin
tend to have higher median popularity compared to other genres, suggesting that songs in these are
generally more popular compared to other genres. Meanwhile, genres like R&B, Rap, and Rock have
lower median popularity but still show significant variance. EDM stands out as having the lowest
overall popularity.

<p align="center">
<img width="635" alt="Screenshot 2025-03-28 at 9 01 06 PM" src="https://github.com/user-attachments/assets/caa69027-e269-45c6-ad14-1f59ecdf1507" />

### Tempo & Danceability

*Figure 4* visualizes the relationship between tempo (measured in BPM) and danceability. The loess
curve added to the plot shows a clear non–linear relationship between the two features. At very low
tempos, danceability remains low, suggesting that in slower songs suited for dancing, the danceability
increases as the tempo rises around 120 BPM. This indicates that tracks with moderate tempos tend
to have higher danceability. As the tempo exceeds 130 BPM, the danceability begins to decline.
This suggests that extremely fast tracks may be less danceable, due to the challenge of maintaining
a danceable rhythm at such high speed.

<p align="center">
<img width="650" alt="Screenshot 2025-03-28 at 9 01 33 PM" src="https://github.com/user-attachments/assets/f72e76b6-3c57-4c99-a283-d6c67d090a28" />

### Tempo & Genres
*Figure 5* shows clear differences in tempo distribution across music genres. EDM, Rap, and Latin
exhibit broader tempo variability, indicating a wide range of song tempos from slow to fast within
these genres. This reflects the diversity in musical styles they encompass. In contrast, genres like
Pop, Rock, and R&B tend to have more consistent tempos, with Pop and Rock centered around
110 BPM and R&B generally having slower, smoother beats below 100 BPM. Overall, there are
clear differences in tempo patterns across genres. Genres like EDM, Rap, and Latin show wider variability, while Pop, Rock, and R&B exhibit more consistent tempo ranges.

<p align="center">
<img width="635" alt="Screenshot 2025-03-28 at 9 02 00 PM" src="https://github.com/user-attachments/assets/c09ab910-93dc-4196-8d87-896e0f96a95e" />

 ## Modeling
To obtain the highest possible expected revenue, we need to build a highly accurate model. Therefore, we will fit and test several different regression models: Linear Regression, CART, Random
Forest, Lasso, Post-Lasso, and Deep Learning. For the modeling we included all numerical columns, genre, sub genre, and year of release from table 1. Additionally, we scaled the track popularity
variable to be between 0 and 1.

### Linear Regression
Our initial approach involved fitting a standard linear regression model without interaction terms.
We included all independent variables available in the dataset to capture any linear relationships
with the target variable. While many variables turned out to be statistically significant, the overall performance of this model was not satisfactory. The lack of interaction terms and nonlinear
transformations likely limited the model’s ability to capture complex patterns in the data.

### Lasso Regression
To address the issue of variable selection and avoid overfitting that arises when we include interaction
effects, we fit a Lasso regression model. Lasso penalizes the absolute size of regression coefficients,
shrinking some coefficients to zero and performing automatic variable selection.
We conducted cross-validation to identify the optimal value of the regularization parameter
λ. This process involved minimizing the mean absolute error (MAE) to determine the best λ.
Ultimately, Lasso selected 344 out of the original set of variables.

### Post-Lasso Regression
After performing variable selection using Lasso, we further refined the model by applying PostLasso regression. In Post-Lasso, we use the variables selected by the Lasso procedure but fit a linear
regression model to these variables, without the regularization term. This step helps mitigate the
bias introduced by the Lasso penalty, as Lasso tends to shrink coefficients, leading to potentially
biased estimates.

### Regression Tree
We used a regression tree to explore how individual features split the data in meaningful ways. By adjusting the complexity parameter, we aimed to avoid overfitting while still capturing key non-linear relationships.

### Random Forest
Building on the regression tree, we trained a Random Forest to boost accuracy and stability. This approach averaged many decision trees, each trained on random subsets of data and features, which helped reduce variance and improve generalization. This model consistently delivered strong performance and highlighted important predictors like acousticness, energy, and valence.

### Deep Learning
For our deep learning model, we implemented a dense feed-forward neural network. After experimenting with different architectures and hyperparameters, we settled on a network with 16 nodes,
using the ReLu activation function to introduce non-linearty in the input layer, two hidden layers,
each with 16 nodes and ReLu activation, and an output layer with a single node and activation
sigmoid to get an output in the 0 to 1 range.

## Evaluation

Upon examining *Figure 6*, we notice the Random Forest model significantly outperformed the other
models in terms of in-sample performance. This substantial improvement raises concerns about
potential overfitting; however, the out of sample performance shows that the Random Forest model
is in fact the top-performing model. Although the performance gap between the Random Forest and other models narrows in this context, it remains the best
option. Based on these findings, we conclude that the Random Forest model is the most effective
predictive model for our dataset and should be utilized for generating recommendations.

<p align="center">
<img width="849" alt="Screenshot 2025-03-28 at 9 02 56 PM" src="https://github.com/user-attachments/assets/35430f76-e636-483a-a9bf-8a6a525d4aec" />

## References

Joakim Arvidsson. Kaggle 30000 Spotify Songs. *https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs*

## Contributors

*Ryan Gilbert*

*Emil Westling*

*Yurou Xu*

*Chelsy Chen*

*Junze Cao*
