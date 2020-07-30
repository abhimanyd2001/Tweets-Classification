# Tweets-Classification
Classifying twitter tweets as viral or non-viral based on certain features.  Analyzing geographical origin of tweets.

## Viral Tweets
`viral_tweets.py` uses the Python SKLearn library's K Neighbors algorithm to use certain features to predict if a tweet will go viral or not.  

## Tweet Locations
`tweet_locations.py` uses natural languange processing to predict where a tweet originated from, out of three major locations: London, Paris, or New York.  The Naive Bayes classifer
was used to calculate the conditional probability of a tweet originating from each of these locations based on the provided JSON files.

The JSON files used in this project came from CodeCademy's 30 day Challenge project database.
