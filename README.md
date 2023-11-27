# capstone_project
Capstone Project

Introduction
	New York City Local Law 18 of 2018, also known as NYC LL 18/2018, was a significant piece of legislation that impacted short-term rental regulations in New York City. The law went into effect on September 7, 2023. The law clarified the definition of short-term rentals. In New York City, it generally referred to rentals of less than 30 consecutive days. Violations of the law could result in substantial fines for both hosts and hosting platforms. The law was aimed, in part, at addressing concerns about the impact of short-term rentals on the availability of affordable housing in the city.

The work to be performed by the team is:
Be able to clarify if there are any visible changes in derived topics for the reviews before the law went into effect and after the law went into effect.
Evaluate listing descriptions to discern how hosts are promoting their spaces in the context of the new guidelines
Categorize listings based on specific features to observe if particular groupings reduce following the law's introduction.

The datasets for 12 months nyc data is located at https://www.dropbox.com/scl/fi/qn2u3exg7kmtmy22ydllv/nyc-airbnb.zip?rlkey=hn9iympw56fpp7p4fcfnzvil1&dl=0

To install requirements: pip -r requirements.txt

Exploratory Analysis
jupyter file name - description of what is being done in file

Supervised Learning
Logistic.ipynb - This logistic model is based on the dataset from August to September. This should help us identify the features of a particular listing that are most likely to be influenced by the law.

Logistic_v2.ipynb - This logistic model notebook is based on the dataset from before August and after September. Our aim is to explore how the features and market structure have changed before and after the introduction of the law.


Topic Modeling
GensimTopicModeling.ipynb - Topics modeled using gensim to compare topics generated pre law and post law.

