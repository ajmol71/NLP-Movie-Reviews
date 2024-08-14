import pandas as pd
import os
import numpy as np
import re
import string
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import random

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords, wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag

import gensim
from gensim.models import Word2Vec, LsiModel, LdaModel, LdaMulticore, TfidfModel, CoherenceModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import dict_from_corpus
from gensim import corpora, similarities

from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import LabelEncoder

import scipy.cluster.hierarchy

from IPython.display import display, HTML

from typing import List, Callable, Dict

import tensorflow_hub as hub
import tensorflow as tf

from dataclasses import dataclass
from collections import defaultdict
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import bertopic
from bertopic import BERTopic

import warnings
warnings.filterwarnings("ignore")

# Utility Functions
def add_movie_descriptor(data: pd.DataFrame, corpus_df: pd.DataFrame):
    """
    Adds "Movie Description" to the supplied dataframe, in the form {Genre}_{P|N}_{Movie Title}_{DocID}
    """
    review = np.where(corpus_df['Review Type (pos or neg)'] == 'Positive', 'P', 'N')
    data['Descriptor'] = corpus_df['Genre of Movie'] + '_' + corpus_df['Movie Title'] + '_' + review + '_' + corpus_df['Doc_ID'].astype(str)

def get_corpus_df(path):
    data = pd.read_csv(path, encoding="utf-8")
    add_movie_descriptor(data, data)
    sorted_data = data.sort_values(['Descriptor'])
    indexed_data = sorted_data.set_index(['Doc_ID'])
    indexed_data['Doc_ID'] = indexed_data.index
    return indexed_data

def remove_punctuation(text):
    return re.sub('[^a-zA-Z]', ' ', str(text))

def lower_case(text):
    return text.lower()

def remove_tags(text):
    return re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)

def remove_special_chars_and_digits(text):
    return re.sub("(\\d|\\W)+"," ", text)

@dataclass
class Document:
    doc_id: str
    text: str

def normalize_document(document: Document) -> Document:
    text = document.text
    text = remove_punctuation(text)
    text = lower_case(text)
    text = remove_tags(text)
    text = remove_special_chars_and_digits(text)

    return Document(document.doc_id, text)

def normalize_documents(documents: List[Document]) -> List[Document]:
    """
    Normalizes text for all given documents.
    Removes punctuation, converts to lower case, removes tags and special characters.
    """
    return [normalize_document(x) for x in documents]

@dataclass
class TokenizedDocument:
    doc_id: str
    tokens: List[str]

def tokenize_document(document: Document) -> TokenizedDocument:
    tokens = nltk.word_tokenize(document.text)
    return TokenizedDocument(document.doc_id, tokens)

def tokenize_documents(documents: List[Document]) -> List[TokenizedDocument]:
    return [tokenize_document(x) for x in documents]

def lemmatize(documents: List[TokenizedDocument]) -> List[TokenizedDocument]:
    result = []
    lemmatizer = WordNetLemmatizer()
    for document in documents:
        output_tokens = [lemmatizer.lemmatize(w) for w in document.tokens]
        result.append(TokenizedDocument(document.doc_id, output_tokens))

    return result

def stem(documents: List[TokenizedDocument]) -> List[TokenizedDocument]:
    result = []
    stemmer = PorterStemmer()
    for document in documents:
        output_tokens = [stemmer.stem(w) for w in document.tokens]
        result.append(TokenizedDocument(document.doc_id, output_tokens))

    return result

def remove_stop_words(documents: List[TokenizedDocument]) -> List[TokenizedDocument]:
    result = []

    stop_words = set(nltk.corpus.stopwords.words('english'))
    for document in documents:
        filtered_tokens = [w for w in document.tokens if not w in stop_words]
        result.append(TokenizedDocument(document.doc_id, filtered_tokens))

    return result

def add_flags(data: pd.DataFrame, casino_royale_doc_ids: List[int], horror_doc_ids: List[int]):
    data['is_casino_royale'] = data.index.isin(casino_royale_doc_ids)
    data['is_horror'] = data.index.isin(horror_doc_ids)

def get_all_tokens(documents: List[TokenizedDocument]) -> List[str]:
    tokens = {y for x in documents for y in x.tokens}
    return sorted(list(tokens))


def clean_method(documents: List[Document]) -> List[TokenizedDocument]:
    """
    Normalizes text, tokenizes, lemmatizes, and removes stop words.
    """
    documents = normalize_documents(documents)
    documents = tokenize_documents(documents)
    documents = lemmatize(documents)
    documents = remove_stop_words(documents)
    documents = stem(documents)

    return documents

def plot_similarity_matrix(data: pd.DataFrame, experiment_name: str, figsize=(25, 25)):
    similarities = cosine_similarity(data, data)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(ax=ax, data=similarities, xticklabels=data.index, yticklabels=data.index);
    #plt.savefig(f'figures/{experiment_name}_heatmap.png')
    plt.close()

def plot_similarity_clustermap(data: pd.DataFrame, experiment_name: str, figsize=(25, 25)):
    similarities = cosine_similarity(data, data)
    cm = sns.clustermap(similarities, metric='cosine', xticklabels=data.index, yticklabels=data.index, method='complete', cmap='RdBu', figsize=figsize)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)
    plt.legend(loc='upper left')
    #plt.savefig(f'figures/{experiment_name}_clustermap.png')
    plt.show()
    plt.close()

def clean_doc(doc):
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    # word stemming
    # ps=PorterStemmer()
    # tokens=[ps.stem(word) for word in tokens]
    return tokens


def plot_tsne(data: pd.DataFrame, perplexity: int, experiment_name: str, figsize=(40, 40)):
    """
    Creates a TSNE plot of the supplied dataframe
    """
    tsne_model = TSNE(perplexity=perplexity, n_components=2, learning_rate='auto', init='pca', n_iter=1000, random_state=32)
    new_values = tsne_model.fit_transform(data)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=figsize)
    labels = list(data.index)
    for i in range(len(x)):
        new_value = new_values[i]
        x = new_value[0]
        y = new_value[1]

        plt.scatter(x, y)
        plt.annotate(labels[i],
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.savefig(f'figures/{experiment_name}_tsne.png')
    plt.show()
    plt.close()

def run_doc2vec(documents: List[TokenizedDocument], embedding_size: int, descriptors_by_doc_ids: Dict[int, str]):
    tagged_documents = [TaggedDocument(document.tokens, [i]) for i, document in enumerate(documents)]
    doc2vec_model = Doc2Vec(tagged_documents, vector_size=embedding_size, window=3, min_count=2, workers=12)

    doc2vec_df = pd.DataFrame()
    for document in documents:
        vector = pd.DataFrame(doc2vec_model.infer_vector(document.tokens)).transpose()
        doc2vec_df = pd.concat([doc2vec_df, vector], axis=0)

    doc2vec_df['Descriptor'] = [descriptors_by_doc_ids[x.doc_id] for x in documents]
    doc2vec_df.set_index(['Descriptor'], inplace=True)
    return doc2vec_df

def run_doc2vec_experiment(documents: List[Document],
                           clean_func: Callable[[List[Document]], List[TokenizedDocument]],
                           embedding_size: int,
                           experiment_name: str):
    cleaned_documents = clean_func(documents)
    doc2vec_df = run_doc2vec(cleaned_documents, embedding_size, descriptors_by_doc_ids)

    plot_similarity_matrix(doc2vec_df, experiment_name)
    plot_similarity_clustermap(doc2vec_df, experiment_name, figsize=(50, 50))
    plot_tsne(doc2vec_df, 30, experiment_name)

def get_elmo_embeddings(texts):
    """
    Generate ELMo embeddings for a list of texts.

    Parameters:
    - texts: List of strings

    Returns:
    - A Tensor containing the ELMo embeddings for the input texts.
    """
    # Convert texts to TensorFlow constants
    text_tf = tf.constant(texts)

    # Generate ELMo embeddings
    embeddings = elmo.signatures['default'](text_tf)['elmo']

    return embeddings

def clusterData( DATA, TRN_DATA, K, TARGET ) :
    print("\n\n\n")
    print("K = ",K)
    print("=======")
    km = KMeans( n_clusters=K, random_state = 1 )
    km.fit( TRN_DATA )
    Y = km.predict( TRN_DATA )
    DATA["CLUSTER"] = Y
    print( DATA.head() )
    G = DATA.groupby("CLUSTER")
    # print( G.mean() )
    print("\n\n\n")
    print( G[ TARGET ].value_counts() )

def run_kmeans(data, doc2vec_input, clusters, target, title):
    kmeans = KMeans(n_clusters= clusters, random_state = 0, algorithm="lloyd").fit(doc2vec_input)
    predictions = kmeans.predict(doc2vec_input)

    silh = silhouette_score(doc2vec_input, predictions)

    cData = data.copy()
    cData["Cluster"] = predictions

    ktitle = "KMeans " + title
    package = [ktitle, predictions, silh, cData]
    # package info:  Title of Run, kmeans predictions, silhouette score, data with cluster column, Grouped by Data
    return package

def run_kmeans_train_test(data, doc2vec_train, doc2vec_test, clusters, target, title):
    kmeans = KMeans(n_clusters= clusters, random_state = 0, algorithm="lloyd").fit(doc2vec_train)
    predictions = kmeans.predict(doc2vec_test)

    # silh = silhouette_score(doc2vec_test, predictions)

    cData = data.copy()
    cData["Cluster"] = predictions

    ktitle = "KMeans " + title
    package = [ktitle, predictions, cData]
    # package info:  Title of Run, kmeans predictions, silhouette score, data with cluster column, Grouped by Data
    return package


# Load Data
CORPUS_PATH = 'https://raw.githubusercontent.com/barrycforever/MSDS_453_NLP/main/MSDS453_ClassCorpus/MSDS453_ClassCorpus_Final_Sec56_v1_20240627.csv'
corpus_df = get_corpus_df(CORPUS_PATH)
documents = [Document(x, y) for x, y in zip(corpus_df.Doc_ID, corpus_df.Text)]

Target = "Genre of Movie"

xDF = corpus_df.copy()
yDF = xDF[Target]
xDF = xDF.drop(Target, axis = 1)

random.seed(1212)
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, train_size = .8, test_size = .2, random_state=1)

# titles_by_doc_ids = {x: y for x, y in zip(xDF['Doc_ID'], corpus_df['Movie Title'])}
# genres_by_doc_ids = {x: y for x, y in zip(corpus_df['Doc_ID'], corpus_df['Genre of Movie'])}
Train_descriptors_by_doc_ids = {x: y for x, y in zip(xTrain['Doc_ID'], xTrain['Descriptor'])}
Test_descriptors_by_doc_ids = {x: y for x, y in zip(xTest['Doc_ID'], xTest['Descriptor'])}

TrainDocuments = [Document(x, y) for x, y in zip(xTrain.Doc_ID, xTrain.Text)]
TestDocuments = [Document(x, y) for x, y in zip(xTest.Doc_ID, xTest.Text)]

# cleaned_tokenized_docs = clean_method(documents)
# texts = [' '.join(x.tokens) for x in cleaned_tokenized_docs]

# doc2vec_exp = run_doc2vec_experiment(documents, clean_func= clean_method, embedding_size = 200, experiment_name= "Corpus_1")  # Doc2Vec Plots
cleaned_train = clean_method(TrainDocuments)
Train_doc2vec_df = run_doc2vec(cleaned_train, embedding_size = 100, descriptors_by_doc_ids = Train_descriptors_by_doc_ids)
# print("Train Doc2Vec\n", Train_doc2vec_df)

cleaned_test = clean_method(TestDocuments)
Test_doc2vec_df = run_doc2vec(cleaned_test, embedding_size = 100, descriptors_by_doc_ids = Test_descriptors_by_doc_ids)
# print("Test Doc2Vec\n", Test_doc2vec_df)


print("Part 1: Clustering ===============")
# K Means Clustering
print("K MEANS TRAINING DATA")
cxTrain = xTrain.copy()
cxTrain[Target] = yTrain

Train_kmeans = []
for i in range(3, 6):
    title = "Train_k" + str(i)
    kmeans = run_kmeans(cxTrain, Train_doc2vec_df, i, Target, title)
    Train_kmeans.append(kmeans)

train_silh = []
for i in Train_kmeans:
    print("Run {}:\n silhouette {}".format(i[0], round(i[2], 4)))
    train_silh.append(i[2])
    G = i[3].groupby("Cluster")
    print(G[Target].value_counts())
    print(G["Review Type (pos or neg)"].value_counts())

for i in range(0, len(train_silh)):
    if train_silh[i] == max(train_silh):
        print("Run {}:".format(Train_kmeans[i][0]))
        G = Train_kmeans[i][3].groupby("Cluster")
        print(G["Descriptor"].value_counts())



print("K MEANS TESTING DATA")
cxTest = xTest.copy()
cxTest[Target] = yTest


# Train kmeans on testing data
# Test_kmeans = []
# for i in range(3, 6):
#     title = "Test_k" + str(i)
#     kmeans = run_kmeans(cxTest, Test_doc2vec_df, i, Target, title)
#     Test_kmeans.append(kmeans)
#
# for i in Test_kmeans:
#     print("Run {}: silhouette {}".format(i[0], round(i[2], 4)))

# Train kmeans on training data, predict testing data
Test_kmeans = []
for i in range(3, 6):
    title = "Tested_k" + str(i)
    kmeans = run_kmeans_train_test(cxTest, Train_doc2vec_df, Test_doc2vec_df, i, Target, title)
    Test_kmeans.append(kmeans)

# test_silh = []
for i in Test_kmeans:
    print("Run {}:".format(i[0]))
    # test_silh.append(i[2])
    G = i[2].groupby("Cluster")
    print(G[Target].value_counts())
    print(G["Review Type (pos or neg)"].value_counts())

# for i in range(0, len(Test_kmeans)):
#     if test_silh[i] == max(test_silh):
#         print("Run {}:".format(Test_kmeans[i][0]))
#         G = Test_kmeans[i][3].groupby("Cluster")
#         for j in G["Descriptor"]:
#             print(j)


print("KMeans Adjustment Model Fix:")
print("Run {}:".format(Test_kmeans[2][0]))
G = Test_kmeans[2][2].groupby("Cluster")
print(G)
for j in G["Descriptor"]:
    print(j)



print("\n============\n")
print("Part 2: Classification ===============")

Encoder = LabelEncoder()
TrainE = Encoder.fit_transform(yTrain)
TestE = Encoder.fit_transform(yTest)


TfIdfVec = TfidfVectorizer(max_features = 4000)
TfIdfVec.fit(xTrain["Text"])

TrainXTF = TfIdfVec.transform(xTrain["Text"])
TestXTF = TfIdfVec.transform(xTest["Text"])

# print(TfIdfVec.vocabulary_)
# print()
# print("TESTXTF:", TestXTF)


# NAIVE BAYES
Naive = naive_bayes.MultinomialNB()
Naive.fit(TrainXTF, TrainE)

predictNB = Naive.predict(TestXTF)
confmatNB = confusion_matrix(predictNB, TestE)
accNB = accuracy_score(predictNB, TestE)

print("Naive Accuracy", round(accNB, 2))
print("NB Confusion Matrix\n", confmatNB)
print()

# SVM
SVM = svm.SVC(C=1.0, kernel="linear", degree=3, gamma="auto")
SVM.fit(TrainXTF, yTrain)

predictSVM = SVM.predict(TestXTF)
print(predictSVM)
confmatSVM = confusion_matrix(predictSVM, yTest)
accSVM = accuracy_score(predictSVM, yTest)

print("SVM Accuracy", round(accSVM, 2))
print("SVM Confusion Matrix\n", confmatSVM)


# BERT Pre-Trained Binary Classification
def bert_classifier_binary(texts, labels):
    # Initialize the sentiment-analysis pipeline
    classifier = pipeline("sentiment-analysis")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.10, random_state=42)

    # Perform sentiment analysis
    results = classifier(X_test, truncation=True)

    # Map BERT's output to binary labels
    predictions = [0 if result['label'] == 'NEGATIVE' else 1 for result in results]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Binary Classification Accuracy: {accuracy}")
    return accuracy

def clean_non_ascii(text):
    # Replace common non-ASCII characters with their ASCII equivalents
    text = text.replace('\x93', '"').replace('\x94', '"')  # Smart double quotes
    text = text.replace('\x91', "'").replace('\x92', "'")  # Smart single quotes
    text = text.replace('\x96', '-').replace('\x97', '-')  # Long dashes

    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

raw_text = [text.strip().lower() for text in corpus_df['Text']]
raw_text_cleaned = [clean_non_ascii(text) for text in raw_text]

xDF = corpus_df.copy()
xDF['processed_text'] = xDF['Text'].apply(lambda x: clean_doc(x))
xDF['raw_text'] = raw_text

# creating final processed text variables for matrix creation
final_processed_text = [' '.join(x) for x in xDF['processed_text'].tolist()]
xDF['final_processed_text'] = final_processed_text
titles = xDF['DSI_Title'].tolist()
processed_text = xDF['processed_text'].tolist()

labels = xDF['Review Type (pos or neg)'].apply(lambda x: 0 if x.lower().split(' ')[0] == 'negative' else 1)

comparisonDF = pd.DataFrame({
    'Label': labels,
    'Processed_Text': xDF['final_processed_text'],
    'Raw_Text': xDF['raw_text']
})
# print(comparisonDF.T)

print(bert_classifier_binary(raw_text, labels))
print(bert_classifier_binary(raw_text_cleaned, labels))
print(bert_classifier_binary(final_processed_text, labels))


# Bert Classifier Multiple
def bert_classifier_multiple_acc(texts, labels):
    # Encode labels
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.10, random_state=42)

    # Load a tokenizer and model suited for your multi-class task
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    #model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(encoder.classes_))
    #model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(encoder.classes_))
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(encoder.classes_))

    # Initialize the pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Perform classification
    results = classifier(X_test, truncation=True)

    # Decode predictions to original labels
    predictions = [int(result['label'].split('_')[-1]) for result in results]
    decoded_predictions = encoder.inverse_transform(predictions)

    # Calculate accuracy
    decoded_y_test = encoder.inverse_transform(y_test)
    accuracy = accuracy_score(decoded_y_test, decoded_predictions)
    print(f"Multi-class Classification Accuracy: {accuracy}")
    return accuracy

print("\nMultiple Bert Classification")
print(bert_classifier_multiple_acc(final_processed_text, xDF["Genre of Movie"]))
print(bert_classifier_multiple_acc(raw_text, xDF["Genre of Movie"]))
print(bert_classifier_multiple_acc(raw_text_cleaned, xDF["Genre of Movie"]))

print("\n Multiple Bert Topic Modeling")
def bert_classifier_multiple(texts, labels):
    # Initialize and fit the label encoder
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(texts, encoded_labels, test_size=0.10, random_state=42)

    # Load a tokenizer and model suited for the multi-class task
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(encoder.classes_))
    #model = AutoModelForSequenceClassification.from_pretrained("bert-large-uncased", num_labels=len(encoder.classes_))

    # Initialize the pipeline
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    # Perform classification
    results = classifier(X_test, truncation=True)

    # Process predictions
    predictions = [int(result['label'].split('_')[-1]) for result in results]

    # Decode predictions and true labels back to original labels
    decoded_predictions = encoder.inverse_transform(predictions)
    decoded_y_test = encoder.inverse_transform(y_test)

    # Return the true labels, predicted labels, and the encoder (for plotting confusion matrix)
    return decoded_y_test, decoded_predictions, encoder

true_labels, predicted_labels, encoder = bert_classifier_multiple(final_processed_text, xDF["Genre of Movie"])
plot_confusion_matrix(true_labels, predicted_labels, encoder.classes_)

def print_confusion_matrix(true_labels, predicted_labels, classes):
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)

    # Create a DataFrame from the confusion matrix
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(cm_df)

print_confusion_matrix(true_labels, predicted_labels, encoder.classes_)


# TOPIC MODELING: LSA and LDA
print("\n============\n")
print("Part 3: Topic Modeling =============")


cleaned_documents = clean_method(documents)
cleaned_document_text = [' '.join(x.tokens) for x in cleaned_documents]

vectorizer = CountVectorizer(ngram_range=(1, 1))
#
transformed_documents = vectorizer.fit_transform(cleaned_document_text)
words = vectorizer.get_feature_names_out()
transformed_documents_as_array = transformed_documents.toarray()

doc_term_freq = pd.DataFrame(transformed_documents_as_array, columns = words, index = corpus_df.Doc_ID)

print(len(cleaned_document_text))
doc_ids = [x for x in doc_term_freq.index]
dic = {}
for i in range(0, len(doc_ids)):
    dic.update({doc_ids[i] : cleaned_document_text[i]})
def create_gensim_lsa_model(doc_clean,number_of_topics,words):

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LSA model
    # train model
    lsamodel = LsiModel(doc_term_matrix
                        ,num_topics=number_of_topics
                        ,id2word = dictionary
                        ,power_iters=100)
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    index = similarities.MatrixSimilarity(lsamodel[doc_term_matrix])

    return lsamodel,dictionary,index

def lsa(tfidf_matrix, terms, n_components = 10):
    #this is a function to execute lsa.  inputs to the function include the tfidf matrix and
    #the desired number of components.

    LSA = TruncatedSVD(n_components=10)
    LSA.fit(tfidf_matrix)

    for i, comp in enumerate(LSA.components_):
        terms_comp = zip(terms, comp)
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
        print("Topic "+str(i)+": ")
        for t in sorted_terms:
            print(t[0])

def create_gensim_lda_model(doc_clean,number_of_topics,words):

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    ldamodel = LdaModel(doc_term_matrix
                        ,num_topics=number_of_topics
                        ,id2word = dictionary
                        ,alpha='auto'
                        ,eta='auto'
                        ,iterations=100
                        ,random_state=23
                        ,passes=20)
    # train model
    print(ldamodel.print_topics(num_topics=number_of_topics, num_words=words))
    index = similarities.MatrixSimilarity(ldamodel[doc_term_matrix])
    return ldamodel,dictionary,index,doc_term_matrix

def lda(tfidf_matrix, terms, topics = 3, num_words = 10):
    #this is a function to perform lda on the tfidf matrix.  function varibales include:
    #tfidf matrix, desired number of topic, and number of words per topic.

    topics = 3
    num_words = 10
    lda = LatentDirichletAllocation(n_components=topics).fit(tfidf_matrix)

    topic_dict = {}
    for topic_num, topic in enumerate(lda.components_):
        topic_dict[topic_num] = " ".join([terms[i]for i in topic.argsort()[:-num_words - 1:-1]])

    print(topic_dict)

def word2vec(processed_text, size = 100):
    #This is a function to generate the word2vec matrix. Input parameters include the
    #tokenized text and matrix size

    #word to vec
    model_w2v = Word2Vec(processed_text, vector_size=300, window=5, min_count=1, workers=4)

    #join all processed DSI words into single list
    processed_text_w2v=[]
    for i in processed_text:
        for k in i:
            processed_text_w2v.append(k)

    #obtian all the unique words from DSI
    w2v_words=list(set(processed_text_w2v))

    #can also use the get_feature_names() from TFIDF to get the list of words
    #w2v_words=Tfidf.get_feature_names()

    #empty dictionary to store words with vectors
    w2v_vectors={}

    #for loop to obtain weights for each word
    for i in w2v_words:
        temp_vec=model_w2v.wv[i]
        w2v_vectors[i]=temp_vec

    #create a final dataframe to view word vectors
    w2v_df=pd.DataFrame(w2v_vectors).transpose()
    print(w2v_df)
    return w2v_df

xDF = corpus_df.copy()
xDF['processed_text'] = xDF['Text'].apply(lambda x: clean_doc(x))

#creating final processed text variables for matrix creation
final_processed_text = [' '.join(x) for x in xDF['processed_text'].tolist()]
titles = xDF['DSI_Title'].tolist()
processed_text = xDF['processed_text'].tolist()

def plot_lsa(processed_text, titles, number_of_topics, words):

    # BARRYC experimental#1
    #model,dictionary,index=create_gensim_lsa_model(processed_text,number_of_topics,words,titles)
    model,dictionary,index=create_gensim_lsa_model(processed_text,number_of_topics,words)

    for doc in processed_text:
        vec_bow = dictionary.doc2bow(doc)
        vec_lsi = model[vec_bow]  # convert the query to LSI space
        sims = index[vec_lsi] # perform a similarity query against the corpus

    fig, ax = plt.subplots(figsize=(30, 10))
    cax = ax.matshow(index, interpolation='nearest')
    ax.grid(True)
    plt.xticks(range(len(processed_text)), titles, rotation=90);
    plt.yticks(range(len(processed_text)), titles);
    fig.colorbar(cax)
    plt.show()
    return model

def plot_lda(processed_text, titles, number_of_topics, words):
    model2, dictionary2, index2, doctermmatrix2 = create_gensim_lda_model(processed_text, number_of_topics, words)

    for doc in processed_text:
        vec_bow2 = dictionary2.doc2bow(doc)
        vec2 = model2[vec_bow2]  # convert the query to embedded space
        sims2 = index2[vec2]  # perform a similarity query against the corpus
        # print(list(enumerate(sims2)))

    fig, ax = plt.subplots(figsize=(30, 10))
    cax = ax.matshow(index2, interpolation='nearest')
    ax.grid(True)
    plt.xticks(range(len(processed_text)), titles, rotation=90);
    plt.yticks(range(len(processed_text)), titles);
    fig.colorbar(cax)
    plt.show()

print("Latent Semantic Analysis:")

LSAmodel, LSAdictionary, LSAindex = create_gensim_lsa_model(processed_text, number_of_topics = 5, words = 10)
print(LSAmodel)
print()
print(LSAdictionary)
print()
print(LSAindex)
plot_lsa(processed_text, titles, 10, 10)

print("Latent Dirichlet Allocation")
LDAmodel = create_gensim_lda_model(processed_text, number_of_topics= 5, words = 10)
plot_lda(processed_text, titles, 10, 10)

