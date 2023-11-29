import pandas as pd
import gzip
import json

# reads a gzip-compressed file line by line and returns each line as a JSON object
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

test = parse('AMAZON_FASHION.json.gz')

# #  creates a DataFrame by converting the JSON data into a dictionary and then into a DataFrame.
# def getDF(path):
#   i = 0
#   df = {}
#   for d in parse(path):
#     df[i] = d
#     i += 1
#   return pd.DataFrame.from_dict(df, orient='index')

# read data from a gzip-compressed JSON file to create DataFrame
# data = getDF('AMAZON_FASHION.json.gz')
# print(data)

# Extract level 0 data
reviews = pd.json_normalize(test)
# Remove blank spaces and non-alphabetic characters from column names
reviews.columns = reviews.columns.str.replace(r'\W+', '', regex=True)
reviews.columns = reviews.columns.str.lower()
pd.set_option('display.max_columns', None)
print(reviews)

# Inspecting the data , Check for any nulls values
reviews.isnull().sum()  
# Inspecting some of the reviews
# https://www.kaggle.com/code/currie32/summarizing-text-with-amazon-reviews
# for i in range(5):
#     print("Review #",i+1)
#     print(reviews.summary[i])
#     print(reviews.reviewtext[i])
#     print()    
    

# summmary of the dataset
reviews.describe()
reviews.shape

reviews['stylecolor'].isna().sum()/len(reviews)*100

# Drop the selected null columns
reviews = reviews.drop(
    columns=['vote','image','stylemetaltype',
             'stylesizename','stylestyle','stylelength','styleteamname',
             'stylestylename','styleformat','stylepackagequantity',
             'stylematerial','styleitemdisplaylength','stylemetaltype',
             'styleitempackagequantity','stylescentname','styleshape',
             'styleshape','stylegemtype'])

# Remove null values
reviews = reviews.dropna() 
    
# check the unique, Remove duplicate rows in place
reviews.drop_duplicates(inplace=True)

# # Plot the distribution of overall scores
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure(figsize=(8, 6))
# sns.histplot(reviews['overall'], kde=True, color='skyblue')
# plt.title('Distribution of overall')
# plt.xlabel('overall scores')
# plt.ylabel('Frequency')
# plt.show()

# # Plot the word card in wordcloud
# from wordcloud import WordCloud
# # Generate a word cloud reviewtext
# reviews['reviewtext']=reviews['reviewtext'].astype('str')
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(reviews['reviewtext']))
# # Display the generated word cloud using matplotlib
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')  # Turn off axis labels
# plt.show()

# Text preparation
import nltk
from nltk.stem import WordNetLemmatizer
# Clean and tokenize
text = pd.DataFrame(reviews['reviewtext'])
# Transform into string object
text['reviewtext'] = text['reviewtext'].astype(str)
# Make text lowercase to remove stopwords
text['reviewtext'] = text['reviewtext'].str.lower()
# Remove non-alphabetic characters
text['reviewtext'] = text['reviewtext'].str.replace(r'\W+', ' ', regex=True)
text['reviewtext'] = text['reviewtext'].str.replace(r'\d', ' ', regex=True)
# Tokenize the text
text['tokens'] = text.apply(lambda row: nltk.word_tokenize(row['reviewtext']), axis=1)
# Tag tokens
text['tokens_tagged'] = text['tokens'].apply(nltk.pos_tag)
# Function to update POS tags in each list
def update_pos_tags(row):
    updated_list = [(token, get_updated_pos(pos)) for token, pos in row]
    return updated_list

# Function to map POS tags to accepted values for lemmatize function
def get_updated_pos(pos):
    if pos.startswith('N'):
        return 'n'  # Noun
    elif pos.startswith('V'):
        return 'v'  # Verb
    elif pos.startswith('R'):
        return 'r'  # Adverb
    elif pos.startswith('J'):
        return 'a'  # Adjective
    else:
        return None  # Return None for other cases

# Apply the update_pos_tags function to the 'tokens_tagged' column
text['tokens_tagged'] = text['tokens_tagged'].apply(update_pos_tags)

# Function to lemmatize tokens
def lemmatize_tokens(tokens_tagged):
    lemmatizer = WordNetLemmatizer()
    lemmatized_list = [(lemmatizer.lemmatize(token, pos=pos_tag) if pos_tag is not None else token) for token, pos_tag in tokens_tagged]
    return lemmatized_list

# Apply the lemmatize_tokens function to the 'token_pos_list' column
text['tokens_lem'] = text['tokens_tagged'].apply(lemmatize_tokens)
# Function to remove stopwords in the English language
def remove_stopwords(tokens_lem):
    # Define stopwords in the English language
    sw = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens_lem if token not in sw]
    return filtered_tokens

text['filtered_tokens'] = text['tokens_lem'].apply(remove_stopwords)
text = text.drop(['reviewtext', 'tokens', 'tokens_tagged', 'tokens_lem'], axis = 1)

# Function to join tokens back into a string if needed
def tokens_to_string(filtered_tokens):
    return ' '.join([token for token in filtered_tokens])
text['text'] = text['filtered_tokens'].apply(tokens_to_string)
print(text['text'])
 
# Create inferences with Mistral 7b
import random
import numpy as np
# Setting the seed for python random numbers
random.seed(13747)
# Setting the seed for numpy-generated random numbers
np.random.seed(37298)
sample_inf = text.sample(3000)
print(sample_inf)
