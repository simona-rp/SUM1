import pandas as pd
# import gzip
import json

# # reads a gzip-compressed file line by line and returns each line as a JSON object
# def parse(path):
#   g = gzip.open(path, 'rb')
#   for l in g:
#     yield json.loads(l)
#
# #  creates a DataFrame by converting the JSON data into a dictionary and then into a DataFrame.
# def getDF(path):
#   i = 0
#   df = {}
#   for d in parse(path):
#     df[i] = d
#     i += 1
#   return pd.DataFrame.from_dict(df, orient='index')
#
# # read data from a gzip-compressed JSON file to create DataFrame
# #data = getDF('AMAZON_FASHION.json.gz')

# Load json file
# Loading a json file keeps the list/dictionary quality, and we can use json_normalize on it
data = []
with open('AMAZON_FASHION.json', 'r') as file:
  for line in file:
    data.append(json.loads(line))

# Extract level 0 data
reviews = pd.json_normalize(data)
reviews.columns = reviews.columns.str.replace(r'\W+', '', regex=True)
reviews.columns = reviews.columns.str.lower()
pd.set_option('display.max_columns', None)
print(reviews)
