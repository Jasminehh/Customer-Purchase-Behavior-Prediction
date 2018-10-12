# Google Store Customer Purchase Behavior Prediction

## Jasmine He

# Introduction
Understanding customer purchase behavior is highly useful for strategic planning and decision-making processes that lead to the company’s future success and growth. This project is to use machine learning models to determine if each customer visiting Google merchandise store is going to make a purchase or not.


# Data Overview
### - Total of 0.9 million visits from 2016-08 to 2017-07
### - Total of 50 different features
### - Main feature categories:
#### 1) Device feature: browser, operating system, isMobile, etc.
#### 2) Source feature: channel, medium, referral path, etc
#### 3) Geography feature: city, country, region, continent, etc
#### 4) Behavior feature: visits, hits, pageviews, bounces, visitStartTime etc
#### 5) Transaction feature: transaction revenue


# Data Preprocessing
## Step 1. Data Cleaning

Here is how the missing data summary look like before cleaning.

![](figures/before_data_clean.png)

Around 10% of the data is missing. After looking at the features with missing data more closely, most of them use NAN to represent 0. The best simplified method is to replace the missing values for those features (bounces, transaction revenue, isTureDirect) with 0.

Notice there are some unnecessary features in our data set, it is optimal to drop those unnecessary features and keep only the useful ones for our experiment.

Below graph shows how clean our data is.

![](figures/after_data_clean.png)

## Step 2. Data Transformation
### 1) Categorize below features:
Browser, Operating System, Source
### 2) Encode categorical below features:
Channel Grouping, Device Category, Continent, Browser Grouping, Operating System Grouping, Source Grouping


# Exploratory Data Analysis
EDA is statisticians way of story telling where you explore data, find patterns and tells insights.

## 1. Channel Distribution
![](figures/channel_dist.png)

## 2. Browser Distribution
![](figures/browser_dist.png)

## 3. Device Distribution
![](figures/device_dist.png)

## 4. Operating System Distribution
![](figures/operatingSystem_dist.png)

## 5. Continent Distribution
![](figures/continent_dist.png)

## 6. Medium Distribution
![](figures/medium_dist.png)


# Machine Learning Modeling

## 1. Feature Selection

### 1). Pair Plot Visualization
![](figures/pairplot.png)

### 2). Pearson Correlation Coefficient Matrix - Original Features
![](figures/coef_all.png)

### 3). Variance Inflation Factors (VIFs)

By running the ReduceVIF function, we can drop the features with vif more than 5 automatically. Below is the screenshot of the output.
![](figures_new/vif.png)


### 4). Pearson Correlation Coefficient Matrix - Reduced Features
![](figures/coef_reduced.png)


## 2. Logistic Regression

### Option 1. Unbalanced Model
![](figures_new/logmodel_unbalanced.png)

### Option 2. Balanced Model
![](figures_new/logmodel_balanced.png)

## 3. Principle Component Analysis
### Option 1. Unbalanced Model
![](figures/PCA.png)

![](figures/PCA_heatmap.png)

### Option 2. Balanced Model
![](figures_new/pca_balanced.png)

![](figures_new/pca_heatmap_balanced.png)


## 4. K-Means Clustering
### Option 1. Unbalanced Model
![](figures/KMeans.png)

### Option 2. Balanced Model
![](figures_new/KMeans_balanced.png)

# Conclusion
![](figures_new/report.png)

Let's compare the accuracy score of Logistic Regression model and the K-Means Clustering model used above.

From the above table, we can see that Logistic Regression on the unbalanced model has the highest precision score (0.94).

# Discussion
