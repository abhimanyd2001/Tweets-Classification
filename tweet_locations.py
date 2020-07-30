import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

new_york_tweets = pd.read_json("new_york.json", lines=True)
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)
print(len(new_york_tweets))
print(new_york_tweets.head())
print(new_york_tweets.columns)
print(new_york_tweets.loc[10]["text"])

print(len(london_tweets))
print(london_tweets.head())
print(london_tweets.columns)
print(london_tweets.loc[10]["text"])

print(len(paris_tweets))
print(paris_tweets.head())
print(paris_tweets.columns)
print(paris_tweets.loc[10]["text"])

# Creating the tweet language dictionary
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()
all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)

# Making the testing and training data set
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size=0.2, random_state=1)

# Creating the count vector
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# Train and test the Naive Bayes Classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(train_counts, train_labels)
predictions = naive_bayes_classifier.predict(test_counts)

print(accuracy_score(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))