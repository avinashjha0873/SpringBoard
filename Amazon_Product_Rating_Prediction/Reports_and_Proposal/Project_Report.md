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

### 1: Removing Special Characters
One important task in text normalization involves removing unnecessary and special characters. These may be special symbols or punctuations that occur in sentences.

Special Characters and symbols are usually non-alphanumeric characters or even occasionally numeric characters(depending on the problem) which adds extra noise to unstructured text data and does not add much significance while analyzing text and utilizing it for feature extraction

```bash
#Defining function to remove special characters keeping only apha characters
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
```bash
#Defined Tokenization fuction
# The following function will take any sentence and convert it into word tokens
# Then strip leading and trailing spaces
def Tokenize_Text(text):
    word_tokens = word_tokenize(text)
    tokens = [token.strip() for token in word_tokens]
    return tokens
```

### 4: Removing Stopwords
Stopwords are the words that has little or no significance especially when consturcting meaningful features from text. They are removed from the text so that we are left with words having maximum significance.

They are usually words that have maximum frequency if you aggregate any corpus of text based on singular tokens.

Ex:- a, the, of and so on.

```bash
#In Python, searching through set is much faster than list.
stopword_set = set(stopwords.words("english"))

#Defining a function to remove stopwords
def Remove_Stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in stopword_set]
    return filtered_tokens
```

### 5: Correcting Words
Incorrect spellings are very normal and also one of the main challenges  faced in Text Normalization. The definition of incorrect here covers words that have spelling mistakes as well as words with several letters repeated that do not contribute much to its overall significance.

#### 5.1: Correcting Repeating Characters
```bash
from nltk.corpus import wordnet

# Define function to remove repeated characters
def Remove_Repeated_Characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if(wordnet.synsets(old_word)):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word) # substitutes a wrong spelling like "Hellooooo" to "Hello"
        return replace(new_word) if new_word != old_word else new_word

    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens
 ```
 
#### 5.2: Correcting Spellings
```bash
from collection import Counter
#Generate a word vocabulary, which will be used as a reference to check the spelling using a file containing severl books from 
#Gutenberg corpus and also list of most frequent words from wiktionary and British National Corpus. You can download it from
# http://norvig.com/big.txt

def tokens(text):
    return re.findall('[a-z]+', text.lower())

path = '../Raw_Data/big.txt'

with open(path) as file:
    doc = file.read()

words = tokens(doc)
word_counts = Counter(words)
```
```bash
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
```bash
# Defining function that returns a subset of words from our candidate set of words obtained from 
# the edit functions, based on whether they occur in our vocabulary dictionary word_counts.
# This gives us a list of valid words from our set of candidate words.
def known(words): 
    "The subset of `words` that appear in the dictionary of word_counts."
    return set(w for w in words if w in word_counts)
```
    
```bash
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
```bash
nlp = spacy.load("en_core_web_sm")
#Defining function for lemmatization
def Lemmatize_Tokens(tokens):
    doc = ' '.join(tokens)
    Lemmatized_tokens = [token.lemma_ for token in nlp(doc)]
    return Lemmatized_tokens
  ```
 ### 7: Text Normalization
 ```bash
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
 
 
