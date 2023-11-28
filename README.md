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

- `Logistic.ipynb`: Analyzes how the features and market structure have changed before and after the introduction of the law.
- `Logistic_v2.ipynb`: Examines how the features and market structure have changed before and after the introduction of the law.
- `GensimTopicModeling.ipynb`: Utilizes Gensim to model topics and compare topics generated before and after the law.
- `notebook_nov02_stats.ipynb`: Conducts statistical analysis.
- `notebook_nov09_viz.ipynb`: Provides visualizations.
- `notebook_nov10_hypo.ipynb`: Conducts hypothesis testing.
- `reviews_topics_over_time.ipynb`: Applies BERTopic on reviews.

## Team Members

- Min Lu
- Shailendra Chalasani
- Weiming Chen

## License

This project is licensed under the [MIT License](LICENSE).
