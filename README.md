# Capstone Project: Analyzing the Impact of NYC Local Law 18 of 2018 on Airbnb Listings

## Introduction

New York City Local Law 18 of 2018, also known as NYC LL 18/2018, was a significant piece of legislation that impacted short-term rental regulations in New York City. The law went into effect on September 7, 2023. The law clarified the definition of short-term rentals in New York City, generally referring to rentals of less than 30 consecutive days. Violations of the law could result in substantial fines for both hosts and hosting platforms. The law was aimed, in part, at addressing concerns about the impact of short-term rentals on the availability of affordable housing in the city.

## Project Goals

The work to be performed by the team includes the following objectives:

1. Clarify if there are any visible changes in derived topics for the reviews before the law went into effect and after the law went into effect.
2. Evaluate the impact of the law on Airbnb listings.
3. Evaluate the geospatial impact of the law.
4. Conduct hypothesis tests on the before and after effects of the law.
5. Categorize listings based on specific features to observe if particular groupings reduce following the law's introduction.
6. Determine feature importance using supervised learning.

## Data Sources

Insideairbnb.com

http://insideairbnb.com/get-the-data/

The datasets used in this analysis are derived from publicly available data on short-term rental listings in New York City, provided by InsideAirbnb. InsideAirbnb is not affiliated with Airbnb and regularly scrapes Airbnb's site to enable transparency.

The data is made available under the Creative Commons CC0 1.0 Universal Public Domain Dedication. As specified in the terms, it can be freely used, modified, and shared by anyone for any purpose.

The raw Airbnb listings data is updated on a regular basis and can be accessed via InsideAirbnb at the following URL:
http://insideairbnb.com/get-the-data.html

Specific data snapshots used in this analysis span November 2022 to October 2023 for New York City.

As the terms permit unrestricted usage, no additional licenses or permissions are needed to replicate this project. Any publications derived from this work should acknowledge InsideAirbnb as the original data source.

This statement clarifies the public domain status of the utilized datasets and refers back to the original provider and terms of use. 

## Datasets

The datasets for this project consist of 12 months of NYC Airbnb data, which can be accessed [here](https://www.dropbox.com/scl/fi/qn2u3exg7kmtmy22ydllv/nyc-airbnb.zip?rlkey=hn9iympw56fpp7p4fcfnzvil1&dl=0).

2022.11 - 2023.10

## Notebook Files

- `Prediction and Feature Analysis.ipynb`: Analyzes the immediate effect for the regulation and prediction
- `Comparative Analysis.ipynb`: Examines how the features and market structure have changed before and after the introduction of the law.
- `GensimTopicModeling.ipynb`: Utilizes Gensim to model topics and compare topics generated before and after the law.
- `notebook_nov02_stats.ipynb`: Conducts statistical analysis.
- `notebook_nov09_viz.ipynb`: Provides visualizations.
- `notebook_nov10_hypo.ipynb`: Conducts hypothesis testing.
- `reviews_topics_over_time.ipynb`: Applies BERTopic on reviews.

### Introduction to Prediction and Feature Analysis.ipynb

Number of listings decreased three time from August to September, and the trend continued from September to October. Our prediction analysis and the immediate effect for the regulation would base on the data from August (before the law) to September (after the law).
![image](https://github.com/shailendra-chalasani/capstone_project/assets/100872992/9185c64e-2683-483e-b789-7d4e3693125e)

Feature preprocessing:  
"instant_bookable_": 1 if there's instant book option, 0 otherwise  
"host_is_superhost_": 1 if the host is superhost, 0 otherwise  
"maximum_nights_30”: We convert the maximum days of stay to a binary variable, with 0 indicating less than a month, and 1 indicating a month or more.  
“host_years”: calculated by how long they have been hosting until 2023   
“price”: logged price has a more symmetrical and balanced distribution.   
“Availability_30”: The number of days of availability in the next 30 days.   
“rating”: Since the Review_scores_rating is highly left skewed, we rank the ratings into three categories, with <4.5 being score 0, 4.5-4.8 being score 1, > 4.8 being score 2.   
“host_listings_count”: The distribution of host listings count is highly right skewed and the most of the host has 5 or less listings. We cap the number of listings to 5.    
“maximum_nights_30”: We convert the maximum days of stay to a binary variable, with 0 indicating less than a month, and 1 indicating a month or more.   
“neighborhood”: The number of listings in Staten Island is too small so we removed it. Then we set Manhattan as our baseline to avoid binary traps.     

Logistic Regression Model:    
Using VIF as a tool to detect multicollinearity, and process False discovery rate to set significance threshold for multiple comparison.

The outcomes of our modeling yield an F1 score of 0.92 for non-dropped listings (label 0) and an F1 score of 0.55 for dropped listings (label 1), with the recall for true positives standing at 0.58.

Feature importance based on the coefficience:
![image](https://github.com/shailendra-chalasani/capstone_project/assets/100872992/32be3c36-f1d2-43e3-9fb0-e37ee93ba107)

Random Forest Model:
The prediction recall score and F1 score are similar to the scores for Logistic model. Below is the feature importance; maximum nights_30, availiability_30, log_prices, host_years, and host_listings_count contribute the most:
![image](https://github.com/shailendra-chalasani/capstone_project/assets/100872992/f1f464e9-8b39-4d7a-84ff-664b7734aa06)


### Introduction to Comparative Analysis.ipynb

We used the dataset prior to July as pre-regulation period. Observing the dropping trend continued from September to October, we classified the dataset after October 1st as post-regulation period.

Initially, we performed a logistic regression analysis on the pre-regulation dataset, employing Variance Inflation Factor (VIF) and False Discovery Rate (FDR) to address multicollinearity and to establish a threshold for comparing multiple features.

Subsequently, we introduced interaction terms to assess the changes of the effect. We applied VIF and FDR again to ensure the robustness of the model and to determine appropriate thresholds for feature comparison.


### Introduction to GensimTopicModeling.ipynb

This notebook performs topic modeling and analysis on online review data before and after the implementation of a new law on September 7, 2023. The goal is to understand if there are any noticeable differences or shifts in the main topics discussed in these online reviews after the law takes effect.

The data preprocessing and modeling workflow includes: tokenizing the review text data, stemming words, removing stop words, appending common bigrams, creating dictionaries and corpus, and finally building LDA (Latent Dirichlet Allocation) models to discover topics. Topics are extracted from both the pre and post-law review data.

The topics and their constituent terms/words are then visualized using heatmaps to allow for easy interpretation and comparison. The Notebook analyzes these heatmaps to explore dominant themes, differences across time periods, and derives insights into the effects of the regulatory change.


### Introduction to reviews_topics_over_time.ipynb

The key research question is - are there observable changes or trends in the prominence of certain topics before versus after a new law enacted on September 7, 2023?

The dataset used contains review text data and associated timestamps. After loading and preparing this data, the BERTopic model from the bertopic library is leveraged to discover topics and analyze their prevalence over the full time period.

The workflow includes:
- Loading review content and timestamps
- Fitting a BERTopic model to the review text corpus
- Transforming the data into extracted topics
- Visualizing topics over time, allowing us to spot rises and declines
- Calling out specific topics and drilling into their temporal patterns

The visualizations will facilitate understanding which discussion themes gain or lose prominence around the time of the law enactment. This can lend insight into whether the law itself may have influenced user sentiment and focus areas.

Additionally, investigating if topic trends align with seasonal holiday peaks can help account for cyclical biases when attributing causality.

### Introduction to notebook_nov02_stats.ipynb

This exploratory data analysis notebook examines Airbnb listing data for New York City before and after the implementation of a new local law on September 7th, 2023. The overarching goal is to analyze the impact of this law by comparing various metrics and patterns in the Airbnb ecosystem before and after the law takes effect.

The key types of analyses conducted include:
- Statistical tests to detect significant differences in metrics like number of listings, prices, availability between the pre and post law periods
- Investigating geographic distribution of listings across boroughs using heatmaps
- Text mining on listing descriptions to identify topic shifts
- Sentiment analysis on review data
- Changes in comply rates and compliance prediction modeling using ML
- Association rule mining to uncover patterns and segments

The data spans multiple months from May 2023 to October 2023, allowing insight into both seasonal effects as well as the law's impact. Preprocessing steps transform the raw data into structured formats amenable for analysis. The notebook attempts to harness both quantitative metrics and unstructured text data to holistically examine the before/after differences and quantify the impact of the regulatory change on Airbnb host behavior and marketplace dynamics. The visualizations and models provide evidence-based assessment of the effects.


### Introduction to notebook_nov09_viz.ipynb

By visualizing various metrics over time at both the city and neighborhood levels, the analysis aims to identify changes that may be attributable to the new policy. Key comparisons and analyses include:
- Geographic distribution of listings across boroughs and neighborhoods using choropleth maps
- Metrics like number of listings, availability, pricing, reviews, and host details
- Segmenting by factors like room types, host attributes, rental durations
- Topic modeling on review content

The notebook leverages libraries like Pandas, Matplotlib, Seaborn, Plotly, and GeoPandas to wrangle the data and create insightful charts. Interactive visualizations are incorporated to allow slicing data along different dimensions.

The choice of visual encodings balances clarity with surface area, providing both summary and detailed perspectives. The notebook format facilitates iterative analysis development.

### Introduction to notebook_nov10_hypo.ipynb

The notebook loads 12 months of Airbnb listing data from November 2022 to October 2023. This allows comparing metrics before and after the law change.

It focuses on using statistical tests like the t-test to quantify if the differences seen in metrics like number of listings, pricing, availability etc. before and after the law are statistically significant or likely occurred by chance.

Some of the hypotheses examined include:
- Whether the drop in number of listings after the law is significant
- If there is a significant change in availability of listings
- Testing for differences across room types and neighborhoods

The analysis calculates the t-statistic and p-values for the metrics, which are then interpreted to accept or reject the hypotheses. Visualizations accompany the statistical tests to provide context.


## Team Members

- Min Lu
- Shailendra Chalasani
- Weiming Chen

## License

This project is licensed under the [MIT License](LICENSE).
