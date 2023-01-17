import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

df = pd.read_csv('reviews.csv')
df.head()

# Convert comment column into a string datatype and standardize it by making it all lower case

df['comments'] = df['comments'].astype('str').str.lower()

# We parse each review by using tokenization and regular expression. We match any word character with one or more occurrences and store it into a Python list into a new column, tokens

regexp = RegexpTokenizer('\w+') # matches any word character (alphanumeric and underscore) with one or more occurrences
df['tokens'] = df['comments'].apply(regexp.tokenize)

# Before proceeding, we get rid of stopwords - these are useless words that do not contribute much meaning to a sentence, such as personal pronouns, articles, auxiliary verbs, conjunctions, etc.

stopwords = nltk.corpus.stopwords.words("english")
df['tokens'] = df['tokens'].apply(lambda x: [item for item in x if item not in stopwords])

# Next, we create a frequency distribution of the tokens

df['tokens_string'] = df['tokens'].apply(lambda x: ' '.join([item for item in x])) # convert list into string

all_words = ' '.join([word for word in df['tokens_string']])
tokenized_words = nltk.tokenize.word_tokenize(all_words) # this is a list
freqdist = FreqDist(tokenized_words)

# Using this dictionary, we keep all tokens that occur more than twice

df['freqdist'] = df['tokens'].apply(lambda x: ' '.join([item for item in x if freqdist[item] > 2]))

# Now, with lemmatization, we group together all the inflected forms of the same word under the same lemma

wordnet_lemma = WordNetLemmatizer()
df['lemma'] = df['freqdist'].apply(wordnet_lemma.lemmatize)

# Finally, we perform sentiment analysis to gauge whether the review is negative, neutral or positive - it is positive if the compound score is closer to 1, negative if it tends to -1, neutral if it is 0.

analyzer = SentimentIntensityAnalyzer()
df['polarity'] = df['lemma'].apply(lambda x: analyzer.polarity_scores(x))

df = pd.concat([df, df['polarity'].apply(pd.Series)], axis=1)

df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >0 else 'neutral' if x==0 else 'negative')

# Finally, we export it to a csv file to save all these modifications

final_df = df.drop(['tokens', 'tokens_string', 'freqdist', 'lemma', 'polarity', 'neg',
       'neu', 'pos', 'compound'], axis=1)

final_df.to_csv('final_reviews.csv', sep=',', header=True, index=False)