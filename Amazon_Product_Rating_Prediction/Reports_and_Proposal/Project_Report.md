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
  
     ![](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/Frequency_of_Ratings2.PNG)
  
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

### 1: Removing Special Characters
One important task in text normalization involves removing unnecessary and special characters. These may be special symbols or punctuations that occur in sentences.

Special Characters and symbols are usually non-alphanumeric characters or even occasionally numeric characters(depending on the problem) which adds extra noise to unstructured text data and does not add much significance while analyzing text and utilizing it for feature extraction

```python
import re

# Defined function will strip leading and trailing spaces 
# Looks for special characters, replace with space
# returning only apha characters

def Remove_Special_Characters(text):
    text = text.strip()
    pattern = '[^a-zA-z]'
    filtered_text = re.sub(pattern, ' ', text) #Replace matches with spaces
    return filtered_text
 ```   
 
### 2: Expanding Contractions
Contractions are words or combinations of words that are shortened by dropping letters or sounds and replacing them with an apostrophe. They exist in either written or spoken forms. In the case of English contractions, they are often created by removing one of the vowels from the word.

Contractions cause various problems with NLP and text analytics like:-
* Don’t and Do Not type of words are treated differently.
* We have a special apostrophe character in the word.

Ideally, we can have a proper mapping for contractions and their corresponding expansions and then use it to expand all the contractions in our text.


    
### 3: Tokenizing Text
Tokenization is the process of transforming a string or document into smaller chunks, which we call tokens. This is usually one step in the process of preparing a text for natural language processing.
 
**Sentence Tokenization** is a process of converting text corpus into sentences which is the first level of tokens. This is also called Sentence Segmentation because we try to segment the text into meaningful sentences.

**Word Tokenization** is a process of splitting sentences into words.
```python
# Defined Tokenization fuction
# The following function will take any sentence and convert it into word tokens
# Then strip leading and trailing spaces

def Tokenize_Text(text):
    word_tokens = word_tokenize(text)
    tokens = [token.strip() for token in word_tokens]
    return tokens
```

### 4: Removing Stopwords
Stopwords are the words that has little or no significance especially when constructing meaningful features from text. They are removed from the text so that we are left with words having maximum significance.

They are usually words that have maximum frequency if you aggregate any corpus of text based on singular tokens.

Ex:- a, the, of and so on.

```python
from nltk.corpus import stopwords

#In Python, searching through set is much faster than list.
stopword_set = set(stopwords.words("english"))

# Defined function will romove remove stopwords
# Here, words not in stopword corpus will be kept

def Remove_Stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in stopword_set]
    return filtered_tokens
```

### 5: Correcting Words
Incorrect spellings are very normal and also one of the main challenges  faced in Text Normalization. The definition of incorrect here covers words that have spelling mistakes as well as words with several letters repeated that do not contribute much to its overall significance.

#### 5.1: Correcting Repeating Characters
```python
from nltk.corpus import wordnet

# Defined function to remove repeated characters

def Remove_Repeated_Characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)') #Regex object to look for repeated charaters
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if(wordnet.synsets(old_word)):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word) # substitutes a wrong spelling like "Hellooooo" to "Hello"
        return replace(new_word) if new_word != old_word else new_word # Replaces repeating characters, till there are none left

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens
 ```
 
#### 5.2: Correcting Spellings
```python
from collections import Counter

#Generate a word vocabulary, which will be used as a reference to check the spelling using a file containing severl books from 
#Gutenberg corpus and also list of most frequent words from wiktionary and British National Corpus. You can download it from
# http://norvig.com/big.txt

def tokens(text):
    return re.findall('[a-z]+', text.lower())

path = '../Raw_Data/big.txt'

with open(path) as file:
    document = file.read()

words = tokens(document)
word_counts = Counter(words)
```

```python
# Define functions that compute sets of words that are one and two edits away from input word.
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
```

```python
# Defining function that returns a subset of words from our candidate set of words obtained from 
# the edit functions, based on whether they occur in our vocabulary dictionary word_counts.
# This gives us a list of valid words from our set of candidate words.

def known(words): 
    "The subset of `words` that appear in the dictionary of word_counts."
    return set(w for w in words if w in word_counts)
```
    
```python
# Define function to correct words
def Correct_Words(words):
    # Get the best correct spellings for the input words
    def candidates(word): 
        # Generate possible spelling corrections for word.
        # Priority is for edit distance 0, then 1, then 2, else defaults to the input word itself.
        candidates = known([word]) or known(edits1(word)) or known(edits2(word)) or [word]
        return candidates
    
    corrected_words = [max(candidates(word), key=word_counts.get) for word in words]
    return corrected_words
```
    
### 6: Lemmatization
The process of lemmatization is to remove word affixes to get to a base form of the word. The base form is also known as the root word, or the lemma will always be present in the dictionary.
```python
import spacy
nlp = spacy.load("en_core_web_sm")

# Defined function for lemmatization

def Lemmatize_Tokens(tokens):
    doc = ' '.join(tokens) # Creates a string doc seperated with spaces 
    Lemmatized_tokens = [token.lemma_ for token in nlp(doc)] # looks for lemma for each word
    return Lemmatized_tokens
  ```
 ### 7: Text Normalization
 ```python
 def Normalize_Text_Corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = text.lower()
        text = Remove_Special_Characters(text)
        tokens = Tokenize_Text(text)
        tokens = Remove_Stopwords(tokens)
        tokens = Remove_Repeated_Characters(tokens)
        tokens = Correct_Words(tokens)
        tokens = Lemmatize_Tokens(tokens)
        text = ' '.join(tokens)
        normalized_corpus.append(text)
    return normalized_corpus
 ```
 After this step, we get Normalized Amazon Reviews.
 
 ## Feature Engineering 
 
Feature Engineering can be defined as a process of making a machine learning algorithm that works well by using domain knowledge to create features in the data set. It is fundamental in the application of machine learning. 

Features can be described as a unique and measurable property for each row or observation in a dataset. Machine learning algorithms usually work with numerical features. If in case they are not numerical, there are various techniques that can be used to deal with such features, like one-hot encoding, imputation, and scaling, etc.

The Vector Space Model, also known as the Term Vector Model, is defined as a mathematical and algebraic model for transforming and representing text documents as numeric vectors of specific terms that form the vector dimensions. It is very useful in case we are dealing with textual data and is very popular in information retrieval and document ranking. 

I will be implementing the following feature-extraction techniques in this project:

* Bag of Words model
* TF-IDF model
* Advance Word Vertorization Models
  * Averaged Word Vectors
  * TF-IDF Weighted Averaged Word Vectors

### 1: Bag of Words Model

The Bag of Words Model, A.K.A BoW is the easiest yet very effective technique to extract features from text data that can be used to train machine learning models.

The approach is very simple, we take a text document and convert them into vectors and each vector represents the frequency of that word in that document. In simple words, a bag-of-words is a representation of text that describes the occurrence of words within a document.

It involves 2 things:-

* The vocabulary of the words.
* A measure of presence of known words.


```python
from sklearn.feature_extraction.text import CountVectorizer

#Defining the fuction to generate bag of words, which take text corpus as input.
#Count Vectorizer, implements both tokenization and occurence counting in a single class

def Bag_of_words(corpus):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(corpus)
    return vertorizer, features
```

### 2: TD-IDF Model

TF-IDF stands for Term Frequency-Inverse Document Frequency.  

  ![](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/TD-IDF%20Formula.PNG)

**Term Frequency** is nothing but what we have computed in the Bag of Words Model ie count of each word in a doc stored in the form of a vector.

**Inverse document frequency** denoted by IDF is the inverse proportion of the number of total Documents and the Number of documents that have that word, on a logarithmic scale.

In my implementation, we will be adding 1 to the document frequency for each term just to indicate that we also have one more document in our corpus that essentially has every term in the vocabulary. This is to prevent potential division-by-zero errors and smoothen the inverse document frequencies. We also add 1 to the result of our IDF computation to avoid ignoring terms completely that might have zero idf.

   ![](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/TD-IDF2.PNG)

where idf(t) represents the idf for the term t, C represents the count of the total number of documents in our corpus, and df(t) represents the frequency of the number of documents in which the term t is present.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

#Defining the function to compute tfidf based feature vectors for documents.

def tfidf(corpus):
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
 ```
 
 ### 3: Average Word Vector
 In this technique, we have a word vector of vocabulary, if token in a sentence is present in the vocabulary we caputre the word. We will sum all the word vectors and devide the result with total number of words matched in the vocabulary to get a finall resulting averaged word vector representation for the text.

![](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/Average_word_Vector_Formula.PNG)

 ```python
 # Define function to average word vectors for a text document
def Average_Word_Vectors(sentence, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features),dtype="float64")
    nwords = 0.
    
    for word in sentence:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model.wv[word])

    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
    return feature_vector
```

```python
# Generalize above function for a corpus of documents  
def Average_Word_Vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    #print(vocabulary)
    features = [Average_Word_Vectors(sentence, model, vocabulary, num_features) for sentence in corpus]
    return np.array(features)
```

### 4: TF-IDF Weighted Average word Vectors
Here we use a technique, strategy of weighing each matched vector with the word TF-TDF score and summing up all the word vectors for a doc and dividing it by the sum of all the TF-IDF loads of the matched words in the document. This would essentially give us a TF-IDF weighted averaged word vector for each document.

![](https://github.com/avinashjha0873/SpringBoard/blob/main/Amazon_Product_Rating_Prediction/Images/TF-IDF%20weighted%20word%20avg%20formula.PNG)

where TWA(D) is the TF-IDF weighted averaged word vector representation for document D, containing words w1, w2, ..., wn, where wv(w) is the word vector representation and tfidf(w) is the TF-IDF weight for the word w.

```python
# Define function to compute tfidf weighted averaged word vector for a document
def tfidf_wtd_avg_word_vectors(words, tfidf_vector, tfidf_vocabulary, model, num_features):
    
    word_tfidfs = [tfidf_vector[0, tfidf_vocabulary.get(word)] 
                   if tfidf_vocabulary.get(word) 
                   else 0 for word in words]    
    word_tfidf_map = {word:tfidf_val for word, tfidf_val in zip(words, word_tfidfs)}
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    vocabulary = set(model.wv.index2word)
    wts = 0.
    for word in words:
        if word in vocabulary: 
            word_vector = model.wv[word]
            weighted_word_vector = word_tfidf_map[word] * word_vector
            wts = wts + word_tfidf_map[word]
            feature_vector = np.add(feature_vector, weighted_word_vector)
    if wts:
        feature_vector = np.divide(feature_vector, wts)
        
    return feature_vector
 ```
 
 ```python
 # Generalize above function for a corpus of documents
def tfidf_weighted_averaged_word_vectorizer(corpus, tfidf_vectors, 
                                   tfidf_vocabulary, model, num_features):
                                       
    docs_tfidfs = [(doc, doc_tfidf) 
                   for doc, doc_tfidf 
                   in zip(corpus, tfidf_vectors)]
    features = [tfidf_wtd_avg_word_vectors(tokenized_sentence, tfidf, tfidf_vocabulary,
                                   model, num_features)
                    for tokenized_sentence, tfidf in docs_tfidfs]
    return np.array(features)
```

## Machine learning and Modeling

Here, I developed models using classification algorithms to predict ratings of products based on the reviews with machine learning. Classification algorithms are supervised ML algorithms that are used to classify data points based on what it has observed in the past. 

This is a three-step process
* Training
* Evaluation 
* Hyperparameter Tuning

Before develop and evaluate models, first, we split data into train and test sets.

### 1: Classification models

There are multiple classification algorithms, but for this project, we will use two algorithms that are quite effective for text classification.

**Logistic Regression(maximum-entropy classification)**
Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic Regression is a binary classifier but can be used for multilevel classification. In this model, the probabilities describing the possible outcomes of a single trial are modeled using a logistic function.

**Random Forest Classifier**
The Random forest classifier creates a set of decision trees from a randomly selected subset of the training set. It is basically a set of decision trees (DT) from a randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction.

In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample) from the training set. In addition, when splitting a node during the construction of the tree, the split that is chosen is no longer the best split among all features. Instead, the split that is picked is the best split among a random subset of the features. As a result of this randomness, the bias of the forest usually slightly increases (with respect to the bias of a single non-random tree) but, due to averaging, its variance also decreases, usually more than compensate for the increase in bias, hence yielding an overall better model.

### 2: Evaluating Classification Models
The performance of classification models is usually based on how well they predict outcomes for new data points.

Several metrics determine a model’s prediction performance, but we will mainly focus on the following metrics:
* Accuracy
* Recall

Accuracy is defined as the overall accuracy or proportion of correct predictions of the model. We have our correct predictions in the numerator divided by all the outcomes in the denominator.

Recall is defined as the number of instances of the positive class that were correctly predicted. This is also known as hit rate, coverage, or sensitivity.  We use the metrics module from scikit-learn, which is very powerful and helps in computing these metrics with a single function.

```python
from sklearn.metrics import accuracy_score, classification_report

def scoring_metrics(true_labels, predicted_labels):
    print ('Accuracy: ', accuracy_score(true_labels,predicted_labels))
    print (classification_report(true_labels, predicted_labels))
```


### 4. Hyperparameter Tuning

**Hyperparameters** are parameters whose values are used to control Learning Process. Hyperparameters are set before the model begins to learn. Different models have different hyperparameters. Like the depth of a decision tree.

Hyper-parameters are parameters that are not directly learned within estimators. In scikit-learn, they are passed as arguments to the constructor of the estimator classes.

In machine learning, **hyperparameter tuning** is the problem of choosing a set of optimal hyperparameters for a learning algorithm. Usually, a metric is chosen to measure the algorithm's performance on an independent data set and hyperparameters that maximize this measure are adopted. Often cross-validation is used to estimate this generalization performance.

There are two ways of Hyperparameter tuning:-
* GridSearchCV, exhaustively considers all parameter combinations
* RandomSearchCV, sample a given number of candidates from a parameter space with a specified distribution

#### 4.1 Logistic Regression with Bag of words features
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'penalty': ['l1', 'l2'], 'solver':['liblinear']} # Declaring Param_Grid

LR_GridSearchCV = GridSearchCV(LogisticRegression(), param_grid = param_grid, cv=5)
LR_GridSearchCV.fit(bow_train_features, train_y)

test_pred = classifier_cv.predict(bow_test_features) 

print("Tuned Parameter: {}".format(classifier_cv.best_params_))
print("Tuned Score: {}".format(classifier_cv.best_score_))
print()

# evaluate model prediction performance 
print ('Test set performance:')
scoring_metrics(true_labels=test_y, predicted_labels=test_pred)
```

#### 4.2 Logistic Regression with TF-IDF Features
```python
from sklearn.model_selection import GridSearchCV

param_grid = {'penalty': ['l1', 'l2'], 'solver':['liblinear']}

classifier_cv = GridSearchCV(LogisticRegression(), param_grid = param_grid, cv=5)
classifier_cv.fit(tfidf_train_features, train_y)

test_pred = classifier_cv.predict(tfidf_test_features) 

print("Tuned Parameter: {}".format(classifier_cv.best_params_))
print("Tuned Score: {}".format(classifier_cv.best_score_))
print()

# evaluate model prediction performance 
print ('Test set performance:')
scoring_metrics(true_labels=test_y, predicted_labels=test_pred)
```

#### 4.3 RandomForestClassifier with Bag of words features
```python
n_options = [10,20,50,100,200]
sample_leaf_options = [1,5,10,50,100,200,500]
param_grid = {'n_estimators': n_options, 'min_samples_leaf': sample_leaf_options}

classifier_cv = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv=5) 
classifier_cv.fit(bow_train_features, train_y)

test_pred = classifier_cv.predict(bow_test_features) 

print("Tuned Parameter: {}".format(classifier_cv.best_params_))
print("Tuned Score: {}".format(classifier_cv.best_score_))
print()

# evaluate model prediction performance 
print ('Test set performance:')
scoring_metrics(true_labels=test_y, predicted_labels=test_pred)
```

#### 4.4 RandomForestClassifier with TF-IDF features
```python
n_options = [10,20,50,100,200]
sample_leaf_options = [1,5,10,50,100,200,500]
param_grid = {'n_estimators': n_options, 'min_samples_leaf': sample_leaf_options}

classifier_cv = GridSearchCV(RandomForestClassifier(), param_grid = param_grid, cv=5) 
classifier_cv.fit(tfidf_train_features, train_y)

test_pred = classifier_cv.predict(tfidf_test_features) 

print("Tuned Parameter: {}".format(classifier_cv.best_params_))
print("Tuned Score: {}".format(classifier_cv.best_score_))
print()

# evaluate model prediction performance 
print ('Test set performance:')
scoring_metrics(true_labels=test_y, predicted_labels=test_pred)
```


## Work Done and Future Work 
In this project, I tried to predict product ratings on Amazon using text reviews. I performed Text Normalization and Feature Engineering to process data and extracted features that would be used in the training process of the models. I trained this model on two different classifiers with 4 different kinds of features. Last, I used hyperparameter tuning to improve the models.
NLP(Natural Language Processing) is a very useful topic. But it is also a CPU-intensive and time-consuming job. I could only make a prototype in this project because of the limitation of computational power and time.

In the future, 
* Full dataset or datasets of other categories can be explored.
* Summary can also be included to train the model
* Other Normalization Techniques, like stemming and Expanding Contractions can be include.
* Other Classification models can be implemented.
* Models can be evaluated using data from other websites
* Use advanced modern NLP tech in text preprocessing.

 ## References
 
 * [How to write Spelling Corrector](https://norvig.com/spell-correct.html)
 * [Scikit Learn Feaure Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
 * [Introduction to Bag of Words](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
 * [Geeks for Geeks](https://www.geeksforgeeks.org/nlp-expand-contractions-in-text-processing/)
