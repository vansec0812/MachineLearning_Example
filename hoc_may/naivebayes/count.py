import json
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag
from collections import Counter
from nltk.corpus import wordnet


# Function to map POS tags to WordNet format
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Function to process the text
import string

# Function to process the text
def convert_todict(text, min_freq=1):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Remove non-ASCII characters and split words
    text = ''.join([char if ord(char) < 128 else ' ' for char in text])  # Remove non-ASCII characters
    words_list = re.split(r'[ ,.(){}:`]+', text)
    
    # POS tagging
    pos_tagged_words = pos_tag([word.lower() for word in words_list])
    
    # Lemmatize words using their POS tags
    normalized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag)) for word, tag in pos_tagged_words
    ]
    
    # Strip punctuation and filter out non-alphabetical words, empty strings, and unwanted characters
    filtered_words = [
        word.strip(string.punctuation) for word in normalized_words
        if word and word.isalpha() and word not in stop_words and not bool(re.search(r'\d', word))
    ]
    
    # Count word frequencies
    word_count = Counter(filtered_words)
    
    # Filter words based on the minimum frequency
    filtered_word_count = {word: count for word, count in word_count.items() if count >= min_freq}

    return filtered_word_count

count_data = {}

# Function to load the data and update count_data
def load_data(file_name, label):
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Process the data to get word counts
        new_data = [convert_todict(entry.get('title', '') + ' ' + entry.get('abstract', '')) for entry in data]
        
        # Add the word counts to the global count_data
        # Combine all word counts for the given label
        combined_word_count = Counter()
        for entry in new_data:
            combined_word_count.update(entry)
        
        # Add combined word count for the label to count_data
        count_data[label] = dict(combined_word_count)

# Load data for different categories and update count_data
load_data('chemistry/chemistry_data.json', 'Chemistry')
load_data('biology/biology_data.json', 'Biology')
load_data('physical/physical_data.json', 'Physics')

# Example: Saving processed chemistry data to a JSON file
with open("count.json", 'w', encoding='utf8') as f:
    json.dump(count_data, f, ensure_ascii=False)
