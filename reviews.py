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
#data = getDF('AMAZON_FASHION.json.gz')
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
for i in range(5):
    print("Review #",i+1)
    print(reviews.summary[i])
    print(reviews.reviewtext[i])
    print()    
    

# summmary of the dataset
reviews.describe()
reviews.shape

# Drop the selected null columns
reviews = reviews.drop(columns=['vote','stylecolor','stylesize','image','stylemetaltype','stylesizename',
                        'stylestyle','stylelength','styleteamname','stylestylename','styleformat','stylepackagequantity','stylematerial','styleitemdisplaylength','stylemetaltype','styleitempackagequantity',
                        'stylescentname','styleshape','styleshape','stylegemtype'])

# Remove null values
reviews = reviews.dropna() 
    
# check the unique, Remove duplicate rows in place
reviews.drop_duplicates(inplace=True)

# Plot the distribution of overall scores
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.histplot(reviews['overall'], kde=True, color='skyblue')
plt.title('Distribution of overall')
plt.xlabel('overall scores')
plt.ylabel('Frequency')
plt.show()

# Calculate the length of the reviewtext and plot the distribution
# Create a new column 'text_length' representing the length of 'text_column'
reviews['reviewtextlength'] = reviews['reviewtext'].apply(len)
plt.figure(figsize=(8, 6))
sns.histplot(reviews['reviewtextlength'], kde=True, color='skyblue')
plt.title('Distribution of Reviewtext length')
plt.xlabel('Number of words')
plt.ylabel('Frequency')
plt.show()

# Plot number of words in box to show the outliers
plt.figure(figsize=(8, 6))
sns.boxplot(x='reviewtextlength', data=reviews, palette="Set2")
plt.title('Box Plot of outliers for reviewtextlength')
plt.show()

# Delete outliers on reviewtextlength
# Define a function to remove outliers based on IQR
def remove_outliers_iqr(data_frame, column):
    Q1 = data_frame[column].quantile(0.25)
    Q3 = data_frame[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data_frame[(data_frame[column] >= lower_bound) & (data_frame[column] <= upper_bound)]

# Specify the column for outlier removal
column_to_check = 'reviewtextlength'

# Remove outliers based on IQR
reviews = remove_outliers_iqr(reviews, column_to_check)
# Plot correlation heatmap
correlation_matrix = reviews.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Plot the word card in wordcloud
from wordcloud import WordCloud
# Generate a word cloud reviewtext
reviews['reviewtext']= reviews['reviewtext'].astype('str')
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(reviews['reviewtext']))
# Display the generated word cloud using matplotlib
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Turn off axis labels
plt.show()
