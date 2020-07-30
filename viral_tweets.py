import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

all_tweets = pd.read_json('random_tweets.json', lines=True)

print(len(all_tweets))
print(all_tweets.columns)

# Tweet text for tweet 1
print(all_tweets.loc[0].text)

# User data from all_tweets
print(all_tweets.loc[0].user)  # is an dictionary representing all the user's info for tweet 1
print(all_tweets.loc[0].user["location"])

# Defining a viral tweet
median = np.median(all_tweets.retweet_count)
print(median)
all_tweets["is_viral"] = np.where(all_tweets.retweet_count > median, 1,
                                  0)  # assigns 1 if the row's retweet count is greater than the median
print(all_tweets[["retweet_count", "is_viral"]].head())

# Making the features
all_tweets["tweet_length"] = all_tweets.apply(lambda tweet: len(tweet.text), axis=1)
all_tweets["followers_count"] = all_tweets.apply(lambda tweet: tweet.user["followers_count"], axis=1)
all_tweets["friends_count"] = all_tweets.apply(lambda tweet: tweet.user["friends_count"], axis=1)
all_tweets["hashtag_count"] = all_tweets.text.str.count("#")
all_tweets["links_count"] = all_tweets.text.str.count("http")
all_tweets["word_count"] = all_tweets.text.str.count(" ") + 1
print(all_tweets.loc[0])

# Normalizing the data
labels = all_tweets["is_viral"]
data = all_tweets[["tweet_length", "word_count", "followers_count", "friends_count"]]
print(labels.head())
scaled_data = scale(data)
print(data.head())
print(scaled_data[:5])

# Creating the training and testing set
training_data, testing_data, training_labels, testing_labels = train_test_split(scaled_data, labels, test_size=0.2,
                                                                                random_state=1)

# Using the classifier
classifier_score = 0
n_neighbors = 0
for i in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(training_data, training_labels)
    if classifier.score(testing_data, testing_labels) > classifier_score:
        classifier_score = classifier.score(testing_data, testing_labels)
        n_neighbors = i

print("The classifier test score is:", classifier_score)

