import pandas as pd
import numpy as np

# Extract annotations
annotations = pd.read_excel('annotations.xlsx')
annotations1 = annotations['annotations_1'].values.tolist()
annotations2 = annotations['annotations_2'].values.tolist()
annotations3 = annotations['annotations_3'].values.tolist()
annotations_all = [annotations1, annotations2, annotations3]

# Embeddings based method
# Vectorize annotations
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec, Doc2Vec
vectorizer_count = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()
# Vectorize based on word count
# count_vectors1 = vectorizer_count.fit_transform(annotations1)
# count_vectors2 = vectorizer_count.fit_transform(annotations2)
# count_vectors3 = vectorizer_count.fit_transform(annotations3)
count_vectors_all = vectorizer_count.fit_transform([" ".join(strings) for strings in annotations_all])

# Vectorire based on word count and importance
# tfidf_vectors1 = vectorizer_tfidf.fit_transform(annotations1)
# tfidf_vectors2 = vectorizer_tfidf.fit_transform(annotations2)
# tfidf_vectors3 = vectorizer_tfidf.fit_transform(annotations3)
tfidf_vectors_all = vectorizer_tfidf.fit_transform([" ".join(strings) for strings in annotations_all])

# Calculate cosine similarity matrices
# Assessment of the Intra-Annotator Similarity: how similar annotations are within each set
from sklearn.metrics.pairwise import cosine_similarity
# cosine_sim_count1 = cosine_similarity(count_vectors1)
# cosine_sim_count2 = cosine_similarity(count_vectors2)
# cosine_sim_count3 = cosine_similarity(count_vectors3)
cosine_sim_count_all = cosine_similarity(tfidf_vectors_all)

# cosine_sim_tfidf1 = cosine_similarity(tfidf_vectors1)
# cosine_sim_tfidf2 = cosine_similarity(tfidf_vectors2)
# cosine_sim_tfidf3 = cosine_similarity(tfidf_vectors3)
cosine_sim_tfidf_all = cosine_similarity(tfidf_vectors_all)

# Flatten matrices into one-dimensional array
# count_flat_cosine1 = cosine_sim_count1.flatten()
# count_flat_cosine2 = cosine_sim_count2.flatten()
# count_flat_cosine3 = cosine_sim_count3.flatten()
count_flat_all = cosine_sim_count_all.flatten()


# tfidf_flat_cosine1 = cosine_sim_tfidf1.flatten()
# tfidf_flat_cosine2 = cosine_sim_tfidf2.flatten()
# tfidf_flat_cosine3 = cosine_sim_tfidf3.flatten()
tfidf_flat_all = cosine_sim_tfidf_all.flatten()

# # Create an agreement matrix
# agreement_matrix = np.vstack((count_flat_cosine1, count_flat_cosine2, count_flat_cosine3)).T
# agreement_matrix2 = np.vstack((tfidf_flat_cosine1, tfidf_flat_cosine2, tfidf_flat_cosine3)).T

# # Calculate Inter-Annotator Similarity
# # Comparison of similarity or dissimilarity of different sets of annotations
# # Pearson's correlation
# pearson_corr = np.corrcoef(agreement_matrix, rowvar=False)
# pearson_corr_coef = pearson_corr[0,1]
# print("Pearson Correlation Coefficient for All Annotations:", pearson_corr_coef)
# # Pearson: 0.4134
# # Moderate correlation

# pearson_corr2 = np.corrcoef(agreement_matrix2, rowvar=False)
# pearson_corr_coef2 = pearson_corr2[0,1]
# print("Pearson Correlation Coefficient for All Annotations:", pearson_corr_coef2)
# # Pearson: 0.4857
# # Moderate correlation

# # Calculate Krippendorff's alpha
# import krippendorff as kd
# # Transpose as krippendorff expects raters as rows and subjects as columns
# arrT = np.array(agreement_matrix).transpose() 
# # Set level of measurement to interval as data is continuous
# alpha_score = kd.alpha(arrT, level_of_measurement='interval')
# print("Krippendorff's Alpha for Inter-Annotator Agreement:", alpha_score)
# # Alpha: 0.4910
# # Moderate agreement

# arrT2 = np.array(agreement_matrix2).transpose() 
# # Set level of measurement to interval as data is continuous
# alpha_score2 = kd.alpha(arrT2, level_of_measurement='interval')
# print("Krippendorff's Alpha for Inter-Annotator Agreement:", alpha_score2)
# # Alpha: 0.5554
# # Moderate agreement

# # Perform one-way ANOVA to calculate ICC
# from scipy.stats import f_oneway
# f_statistic, p_value = f_oneway(flat_cosine1, flat_cosine2, flat_cosine3)
# # p-value < 0.05
# icc = 1 - (f_statistic / (f_statistic + (len(flat_cosine1) - 1)))
# print("Intra-Class Correlation (ICC):", icc)
# # ICC: 0.9741
# # Almost perfect agreement

# f_statistic2, p_value2 = f_oneway(flat_cosine11, flat_cosine22, flat_cosine33)
# # p-value < 0.05
# icc2 = 1 - (f_statistic / (f_statistic + (len(flat_cosine1) - 1)))
# print("Intra-Class Correlation (ICC):", icc2)
# # ICC: 0.9881
# # Almost perfect agreement

# Bleu score
import nltk
from nltk.translate.bleu_score import sentence_bleu
# Tokenize annotattions
tokens1 = pd.DataFrame(annotations1, columns=['summary'])
tokens1['summary'] = tokens1['summary'].replace(r'\W+', ' ', regex=True)
tokens1 = tokens1.apply(lambda row: nltk.word_tokenize(row['summary']), axis = 1)
tokens1 = [' '.join(token) for token in tokens1]

tokens2 = pd.DataFrame(annotations2, columns=['summary'])
tokens2['summary'] = tokens2['summary'].replace(r'\W+', ' ', regex=True)
tokens2 = tokens2.apply(lambda row: nltk.word_tokenize(row['summary']), axis = 1)
tokens2 = [' '.join(token) for token in tokens2]

tokens3 = pd.DataFrame(annotations3, columns=['summary'])
tokens3['summary'] = tokens3['summary'].replace(r'\W+', ' ', regex=True)
tokens3 = tokens3.apply(lambda row: nltk.word_tokenize(row['summary']), axis = 1)
tokens3 = [' '.join(token) for token in tokens3]

bleu_score = sentence_bleu(tokens1, tokens3, weights=(0.5, 0.5, 0, 0))
print("BLEU Score:", bleu_score)