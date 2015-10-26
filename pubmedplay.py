from Bio import Entrez
import json
import numpy as np
import pandas as pd


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import decomposition
from sklearn.cluster.bicluster import SpectralCoclustering

def fetch_details(id_list):
    ids = ','.join(id_list)
    Entrez.email = 'your.email@example.com'
    handle = Entrez.efetch(db='pubmed',
                           retmode='xml',
                           id=ids)
    results = Entrez.read(handle)
    return results

def search(query,N):
    Entrez.email = 'your.email@example.com'
    handle = Entrez.esearch(db='pubmed',
                            sort='relevance',
                            retmax=N,
                            retmode='xml',
                            term=query)
    results = Entrez.read(handle)
    return results

def mergeAbstract(abstract):
    return("".join(abstract))

results = search('zebrafish',500)
id_list = results['IdList']
papers = fetch_details(id_list)

listOfTitles = [paper['MedlineCitation']['Article']['ArticleTitle'] for paper in papers]

listOfAbstracts = []
for paper in papers:
    if 'Abstract' in paper['MedlineCitation']['Article'].keys():
        listOfAbstracts.append(mergeAbstract(paper['MedlineCitation']['Article']['Abstract']['AbstractText']))

# Create TF-IDF matrix
vect = TfidfVectorizer(max_df = 1)
tfidf = vect.fit_transform(listOfAbstracts)



# Non-negative Matrix Factorization
num_topics = 2
num_top_words = 5
nmf = decomposition.NMF(n_components=num_topics, random_state=1)
doctopic = nmf.fit_transform(tfidf)
topic_words = []
vocab = np.array(vect.get_feature_names())

for topic in nmf.components_:
    word_idx = np.argsort(topic)[::-1][0:num_top_words]
    topic_words.append([vocab[i] for i in word_idx])

# Coclustering
cocluster = SpectralCoclustering(n_clusters=5,svd_method='arpack', random_state=0)
cocluster.fit(tfidf)
y_cocluster = cocluster.row_labels_
x_cocluster = cocluster.column_labels_

# print(np.array(vect.get_feature_names())[x_cocluster == 4])
