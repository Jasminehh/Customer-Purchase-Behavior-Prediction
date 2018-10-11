# Google Merchandise Store Customer Purchase Behavior Prediction

# Introduction
## Understanding customer purchase behavior is highly useful for strategic planning and decision-making processes that lead to the company’s future success and growth.

## This project is to use machine learning models to determine if each customer visiting Google merchandise store is going to make a purchase or not.


# Data Overview
### - Total of 0.9 million visits from August,2016 to July,2017
### - 50 different features
### - Main feature categories:
#### - Device feature: browser, operating system, isMobile, etc.
#### - Source feature: channel, medium, referral path, etc
#### - Geography feature: city, country, region, continent, etc
#### - Behavior feature: visits, hits, pageviews, bounces, visitStartTime etc
#### - Transaction feature: transaction revenue


<img src=‘figures/test.png’/>


# Methodology

## To predict which prospects are ready to make their first purchase, a likelihood to buy model evaluates non-transaction customer data, such as how many times a customer clicked on an email or how the customer interacts with your website. These models can also take into account certain demographic data.

## For example, in consumer marketing they may compare gender, age, and zip code to other likely buyers. In business marketing, relevant demographics may include industry, job title, and geography.

# Data Preprocessing
## Data Cleaning

## - Handle Missing Data Summary

![](figures/before_data_clean.png)




Around 10% of the data is missing. After looking at the features with missing data more closely, most of them use na to represent 0. Let us replace the missing values for those features (bounces, transaction revenue, isTureDirect) with 0.

drop unnecessary features and keep only the useful ones for our experiment.

<img src=‘figures/after_data_clean.png’/>

## Data Transformation
## - Categorize feature -- Browser, Operating System, Source
## - Encode categorical feature columns -- Channel Grouping, Device Category, Continent, Browser Grouping, Operating System Grouping, Source Grouping


# Exploratory Data Analysis
## EDA is statisticians way of story telling where you explore data, find patterns and tells insights.

## - Channel Distribution
<img src=‘figures/channel_dist.png’/>

## - Browser Distribution
<img src=‘figures/browser_dist.png’/>

## - Device Distribution
<img src=‘figures/browser_dist.png’/>

## - Operating System Distribution
<img src=‘figures/operatingSystem_dist.png’/>

## - Continent Distribution
<img src=‘figures/continent_dist.png’/>

## - Medium Distribution
<img src=‘figures/medium_dist.png’/>


# Machine Learning Modeling

## Feature Selection

## - Pairplot Visulization

<img src=‘figures/pairplot.png’/>

## - Pearson Correlation Coeficient Matrix

<img src=‘figures/coef_all.png’/>

## - Variance Inflation Factors (VIFs)

????? need a vif pic

<img src=‘figures/coef_reduced.png’/>


## Logistic Regression

<img src=‘figures/ROC.png’/>

????? need an accuracy report

## Confusion Matrix

<img src=‘figures/confusion_matrics.png’/>

## Principle Component Analysis

<img src=‘figures/PCA.png’/>

<img src=‘figures/PCA_heatmap.png’/>

## K-Means Clustering

<img src=‘figures/KMeans.png’/>


## Model Comparison

### Let's compare the accuracy score of Logistic Regression model and the K-Means Clustering model used above.

### From the above table, we can see that has the highest accuracy score.

### Among these two, we choose Support Vector Machines classifier as it has the ability to limit overfitting as compared to Decision Tree classifier.
