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

## Datasets

The datasets for this project consist of 12 months of NYC Airbnb data, which can be accessed [here](https://www.dropbox.com/scl/fi/qn2u3exg7kmtmy22ydllv/nyc-airbnb.zip?rlkey=hn9iympw56fpp7p4fcfnzvil1&dl=0).

## Notebook Files

- `Prediction and immediate effect analysis.ipynb`: Analyzes the immediate effect for the regulation and prediction
- `Comparative Analysis.ipynb`: Examines how the features and market structure have changed before and after the introduction of the law.
- `GensimTopicModeling.ipynb`: Utilizes Gensim to model topics and compare topics generated before and after the law.
- `notebook_nov02_stats.ipynb`: Conducts statistical analysis.
- `notebook_nov09_viz.ipynb`: Provides visualizations.
- `notebook_nov10_hypo.ipynb`: Conducts hypothesis testing.
- `reviews_topics_over_time.ipynb`: Applies BERTopic on reviews.

## Introduction to Prediction and immediate effect analysis.ipynb

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


## Introduction to Comparative Analysis.ipynb

We used the dataset prior to July as pre-regulation period. Observing the dropping trend continued from September to October, we classified the dataset after October 1st as post-regulation period.

Initially, we performed a logistic regression analysis on the pre-regulation dataset, employing Variance Inflation Factor (VIF) and False Discovery Rate (FDR) to address multicollinearity and to establish a threshold for comparing multiple features.

Subsequently, we introduced interaction terms to assess the changes of the effect. We applied VIF and FDR again to ensure the robustness of the model and to determine appropriate thresholds for feature comparison.







## Team Members

- Min Lu
- Shailendra Chalasani
- Weiming Chen

## License

This project is licensed under the [MIT License](LICENSE).
