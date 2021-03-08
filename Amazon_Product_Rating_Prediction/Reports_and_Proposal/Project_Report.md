# Amazon Products Rating Prediction

## Introduction
These days, you can find reviews by users on almost every product or event. Other than offering significant information, these reviews unequivocally sway the purchase choice of users. Numerous shoppers are adequately impacted by online surveys when settling on their buy choices. Depending on online surveys has subsequently become natural for buyers.
When looking for a product a customer needs to find useful and credible reviews as fast as possible and comparing text reviews can be a daunting task. So Amazon has 5 stars rating which will give you an overview of the quality of the product with 1 meaning not that good and 5 meaning really good.

However, many other platforms do not have this rating system. In such cases, it is very important to have a model that could predict the rating of the product using text reviews.

The purpose of this project is to develop a model that could predict its rating based on text reviews. While this model is built to work with any kind of product, this project includes reviews from Amazon’s phones and accessories dataset.

## About the Dataset
I used Amazon Products data, which contains product reviews and metadata from Amazon, including 142.8 million reviews spanning May 1996 ~ July 2014. The dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs).

In this project, I use a 5-core dataset of Cell Phones and Accessories, which is a subset of the data in which all users and items have at least 5 reviews. [Link to the Dataset](http://jmcauley.ucsd.edu/data/amazon/)


Sample review is as follows:
```bash
"reviewerID": "A2SUAM1J3GNN3B",  
"asin": "0000013714",  
"reviewerName": "J. McDonald",  
"helpful": [2, 3],  
"reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",  
"overall": 5.0,  
"summary": "Heavenly Highway Hymns",  
"unixReviewTime": 1252800000,  
"reviewTime": "09 13, 2009"  
```

## Preliminary Analysis

* Summary of the dataset

  ![Summary of Dataset](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/Preliminary_Analysis.PNG)

* Distribution of Rating Score

  ![Distribution of Rating Score](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/Frequency_of_Ratings.PNG)
  
## Pre-processing - Text Normalization
Preprocessing is an important step when working with Natural Language Processing(NLP) and text analysis. Normally, text data or corpus are not well-formatted in their raw format. Therefore we need to use various techniques to transform that data into a suitable format. Machine Learning Algorithm usually works with features that are numerical which can be obtained by cleaning, normalizing, and pre-processing the raw text data.

Text Normalization(Text wrangling or cleansing) is a process of transforming text into a form that can be used by NLP and analytics systems and applications as input. It can 
consist of various steps like:-

* Cleaning Text
* Tokenizing Text
* Removing Special Characters
* Expanding Contractions
* Case Conversions
* Removing Stopwords
* Correcting Words
* Stemming
* Lemmatization

We will use most of the techniques in this project.

### Tokenizing Text
Tokenization is the process of transforming a string or document into smaller chunks, which we call tokens. This is usually one step in the process of preparing a text for natural language processing.
 
Sentence Tokenization is a process of converting text corpus into sentences which is the first level of tokens. This is also called Sentence Segmentation because we try to segment the text into meaningful sentences.

Word Tokenization is a process of splitting sentences into words.

