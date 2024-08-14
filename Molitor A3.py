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
import pkg_resources

from sklearn import model_selection, naive_bayes, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, r2_score, silhouette_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.manifold import TSNE, MDS
from sklearn.preprocessing import LabelEncoder

import scipy.cluster.hierarchy
from timeit import default_timer as timer

from IPython.display import display, HTML

from typing import List, Callable, Dict, Tuple, Set

import tensorflow_hub as hub
import tensorflow as tf

import networkx as nx
from tqdm import tqdm

from dataclasses import dataclass
from collections import defaultdict, Counter
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

import bertopic
from bertopic import BERTopic

import spacy as sp
import spacy_llm
from spacy.matcher import Matcher
from spacy.tokens import Span
import huggingface
import huggingface_hub

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

nlp = sp.load("en_core_web_lg")

# Functions


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

def get_sentences(text: str) -> List[str]:
    return [str(x) for x in nlp(text).sents]

def get_coref_resolved_sentences(text: str) -> List[str]:
    return [str(x) for x in nlp(text).sents]

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


def run_tfidf(documents: List[Document],
              clean_func: Callable[[List[Document]], List[TokenizedDocument]],
              important_prevalent_terms: List[str],
              experiment_name: str,
              dataframe: pd.DataFrame(),
              output_tfidf_vectors: bool = False,
              output_vocabulary: bool = True):
    cleaned_documents = clean_func(documents)
    cleaned_document_text = [' '.join(x.tokens) for x in cleaned_documents]

    vectorizer = TfidfVectorizer(use_idf=True,
                                 ngram_range=(1, 1),
                                 norm=None)

    transformed_documents = vectorizer.fit_transform(cleaned_document_text)
    transformed_documents_as_array = transformed_documents.toarray()

    output_dir = f'output/{experiment_name}_Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_tfidf_vectors:
        for counter, doc in enumerate(transformed_documents_as_array):
            tf_idf_tuples = list(zip(vectorizer.get_feature_names_out(), doc))
            one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'score']) \
                .sort_values(by='score', ascending=False) \
                .reset_index(drop=True)

            one_doc_as_df.to_csv(f'{output_dir}/{corpus_df["Submission File Name"][counter]}')

    if output_vocabulary:
        with open(f'{output_dir}/vocabulary.txt', 'w') as vocab:
            words = sorted(vectorizer.get_feature_names_out())
            print('\n'.join(words), file=vocab)

    # Create document-term dataframe
    doc_term_matrix = transformed_documents.todense()
    doc_term_df = pd.DataFrame(doc_term_matrix,
                               columns=vectorizer.get_feature_names_out(),
                               index=dataframe.Doc_ID)

    # Print the top 10 mean TF-IDF values
    top10_tfidf = pd.DataFrame(doc_term_df.mean().sort_values(ascending=False).head(10))
    top10_tfidf.rename(columns={0: 'Mean TF-IDF'}, inplace=True)
    display(top10_tfidf)

    # Print DF with Prevalent Terms
    prev_tfidf = pd.DataFrame(doc_term_df[important_prevalent_terms].mean().sort_values(ascending=False).head(10))
    prev_tfidf.rename(columns={0: 'Mean TF-IDF Prevalent'}, inplace=True)
    display(prev_tfidf)

    # Collect result into a dataframe
    tfidf_results = pd.DataFrame(index=important_prevalent_terms)

    all_tfidf_results = doc_term_df[[x for x in important_prevalent_terms if x in doc_term_df.columns]].mean().round(2)
    tfidf_results['All Movies'] = all_tfidf_results

    # Visuals

    plt.hist(doc_term_df.mean(), 100, range=(0, 8))

    print(f'Mean Mean: {doc_term_df.mean().mean()}')
    print(f'Vocabulary size: {doc_term_df.shape[1]}')

    descriptors = TargetDF['Descriptor']

    target_ids = [x for x in TargetDF["Doc_ID"]]

    similarities = cosine_similarity(doc_term_df.loc[target_ids], doc_term_df.loc[target_ids])
    fig, ax = plt.subplots(figsize=(20, 20))
    labels = [descriptors_by_doc_ids[x.doc_id] for x in TarDocuments]
    sns.heatmap(ax=ax, data=similarities, xticklabels=labels, yticklabels=labels)
    # plt.savefig(f'figures/{experiment_name}_heatmap_documents.png')
    plt.show()

important_prevalent_terms = [
    'film',
    'best',
    'actor',
    'charact'
]

# run_tfidf(TarDocuments, clean_method, important_prevalent_terms, 'TFIDF_Target', TargetDF)

def get_word2vec_vectors(documents: List[TokenizedDocument], embedding_size: int) -> pd.DataFrame:
    tokens = [x.tokens for x in documents]

    word2vec_model = Word2Vec(sentences=tokens, vector_size=embedding_size, window=3, min_count=1, workers=12)

    vectors = {}
    for i in word2vec_model.wv.index_to_key:
        temp_vec = word2vec_model.wv[i]
        vectors[i] = temp_vec

    result = pd.DataFrame(vectors).transpose()
    result = result.sort_index()
    return result


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

def run_word2vec_experiment(documents: List[Document],
                            clean_func: Callable[[List[Document]], List[TokenizedDocument]],
                            embedding_size: int,
                            chosen_tokens: List[str],
                            experiment_name: str):
    cleaned_documents = clean_func(documents)

    word2vec_df = get_word2vec_vectors(cleaned_documents, embedding_size)
    filtered_word2vec_df = word2vec_df.loc[chosen_tokens].copy()

    plot_tsne(filtered_word2vec_df, 30, experiment_name)
    plot_similarity_matrix(filtered_word2vec_df, experiment_name)
    plot_similarity_clustermap(filtered_word2vec_df, experiment_name)

default_stopwords=\
set(nltk.corpus.stopwords.words('english')).union(set(nlp.Defaults.stop_words)).union({' ', ''})

# Load Data
CORPUS_PATH = 'https://raw.githubusercontent.com/barrycforever/MSDS_453_NLP/main/MSDS453_ClassCorpus/MSDS453_ClassCorpus_Final_Sec56_v1_20240627.csv'
corpus_df = get_corpus_df(CORPUS_PATH)
documents = [Document(x, y) for x, y in zip(corpus_df.Doc_ID, corpus_df.Text)]
documents = clean_method(documents)   # tokenized
Encoder = LabelEncoder()
ETarget = Encoder.fit_transform(corpus_df["Genre of Movie"])
w2v = get_word2vec_vectors(documents, embedding_size=200)
print(w2v[1:5])

print(ETarget)
Target = "Genre of Movie"

xDF = corpus_df.copy()
yDF = xDF[Target]
xDF = xDF.drop(Target, axis = 1)
xDF['raw_sentences'] = corpus_df.Text.apply(get_sentences)
xDF['processed_text'] = xDF['Text'].apply(lambda x: clean_doc(x))

#creating final processed text variables for matrix creation
final_processed_text = [' '.join(x) for x in xDF['processed_text'].tolist()]
titles = xDF['DSI_Title'].tolist()
processed_text = xDF['processed_text'].tolist()



# Knowledge Graph Functions
def map_edges(map_to: str, map_from: Set[str], df: pd.DataFrame):
    print(f'Before mapping {", ".join(map_from)} -> {map_to}: {sum(df.edge == map_to)}')
    df['edge'] = np.where(kg_df.edge.isin(map_from), map_to, kg_df.edge)
    print(f'After mapping {", ".join(map_from)} -> {map_to}: {sum(df.edge == map_to)}')

def map_sources_and_targets(map_to: str, map_from: Set[str], df: pd.DataFrame):
    before = sum(df.source == map_to) + sum(df.target == map_to)
    print(f'Before mapping {", ".join(map_from)} -> {map_to}: {before}')

    df['source'] = np.where(kg_df.source.isin(map_from), map_to, kg_df.source)
    df['target'] = np.where(kg_df.target.isin(map_from), map_to, kg_df.target)

    after = sum(df.source == map_to) + sum(df.target == map_to)
    print(f'After mapping {", ".join(map_from)} -> {map_to}: {after}')

def get_neighborhood(sources: Set[str], edge_types: Set[str], depth: int, df: pd.DataFrame) -> pd.DataFrame:
    output = []

    for d in range(depth):
        if edge_types is not None:
            rows = df[(df.edge.isin(edge_types)) & ((df.source.isin(sources)) | (df.target.isin(sources)))].copy()
        else:
            rows = df[(df.source.isin(sources)) | (df.target.isin(sources))].copy()

        output.append(rows)
        sources = set(rows.target).union(set(rows.source))

    return pd.concat(output).drop_duplicates()

def find_sources_and_targets_with_patterns(patterns: List[str], df: pd.DataFrame):
    mask = np.zeros(kg_df.shape[0])
    for pattern in patterns:
        mask = mask | (df.source.str.contains(pattern)) | (df.target.str.contains(pattern))

    return df[mask]

def plot_graph(df: pd.DataFrame, show_edges: bool = False, figsize: Tuple[int, int] = (12, 12), use_circular: bool=True):
    graph = nx.from_pandas_edgelist(df, "source", "target", edge_attr='edge', create_using=nx.MultiDiGraph())

    plt.figure(figsize=figsize)
    if use_circular:
        pos = nx.circular_layout(graph)
    else:
        pos = nx.kamada_kawai_layout(graph)

    nx.draw(graph, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    if show_edges:
        nx.draw_networkx_edge_labels(graph, pos=pos, font_size=8)

    plt.show()

def get_top_sources_and_targets(df: pd.DataFrame, top: int = 10):
    return (Counter(df.source) + Counter(df.target)).most_common(top)

def get_top_edges(df: pd.DataFrame, top: int = 10):
    return Counter(df.edge).most_common(top)

def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.10, test_split=0.10):
       # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=12)
    indices_or_sections = [int(.8*len(df)), int(.9*len(df))]
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    return train_ds, val_ds, test_ds

# Entity Extraction Functions
def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""

  for tok in nlp(sent):
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text

      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text

      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text

      ## chunk 5
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text

  return [ent1.strip(), ent2.strip()]

def get_relation(sent):
    try:
        doc = nlp(sent)

        # Matcher class object
        matcher = Matcher(nlp.vocab)

        #define the pattern
        pattern = [{'DEP':'ROOT'},
                {'DEP':'prep','OP':"?"},
                {'DEP':'agent','OP':"?"},
                {'POS':'ADJ','OP':"?"}]
        matcher.add("matching_1", [pattern])
        matches = matcher(doc)
        k = len(matches) - 1
        span = doc[matches[k][1]:matches[k][2]]

        return(span.text)
    except:
        pass

def get_subject_verb_object(sent):
  ent1 = ""
  ent2 = ""
  root = ""

  for tok in nlp(sent):
      if tok.dep_ == 'ROOT':
        root = tok.text
      elif tok.dep_ == "nsubj":
        ent1 = tok.text
      elif tok.dep_ == "dobj":
        ent2 = tok.text

      if ent1 != '' and ent2 != '' and root != '':
        break

  return [ent1, root, ent2]

# Visualization Functions
def plot_confusion_matrix_labeled(y_true, y_pred, CLASSES_LIST):
    mtx = confusion_matrix(y_true, y_pred)
    # define classes
    classes = CLASSES_LIST
    temp_df = pd.DataFrame(data=mtx,columns=classes)
    temp_df.index = classes
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(temp_df, annot=True, fmt='d', linewidths=.75,  cbar=False, ax=ax,cmap='Blues',linecolor='white')
    #  square=True,
    plt.ylabel('true label')
    plt.xlabel('predicted label')

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

def display_training_curves(training, validation, title, subplot):
  ax = plt.subplot(subplot)
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])

movie_df = xDF[xDF['Movie Title'] == 'The_Toxic_Avenger'].copy()
CorpusSentences = [y for x in movie_df.raw_sentences for y in x]
entity_pairs = [get_entities(x) for x in tqdm(CorpusSentences)]

# print(CorpusSentences[1:5])
# print(entity_pairs[1:5])

relations = [get_relation(x) for x in CorpusSentences]
# print(relations[1:5])

# #extract subject and object
source = [i[0] for i in entity_pairs]
target = [i[1] for i in entity_pairs]
kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
#
# # Move everything to lower case
kg_df.source = kg_df.source.str.lower()
kg_df.target = kg_df.target.str.lower()
kg_df.edge = kg_df.edge.str.lower()
#
# # Filter out empties
kg_df = kg_df[kg_df.source != '']
kg_df = kg_df[kg_df.target != '']
kg_df = kg_df[kg_df.edge != ''].copy()

print(kg_df.head().T)


plot_graph(kg_df, use_circular = True)



# BERT NER --- Spacy + LLM
def setup_ner_pipeline(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    """
    Set up the NER pipeline using the specified BERT model.
    """
    try:
        ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
        print("NER pipeline set up successfully.")
        return ner_pipeline
    except Exception as e:
        print(f"An error occurred while setting up the NER pipeline: {e}")
        return None

def perform_ner_and_store_in_df(data, ner_pipeline):
    """
    Perform NER on the loaded dataset using the specified NER pipeline and store the results in DataFrames.
    The dataset should have 'title' and 'content' columns.
    """
    if data is not None and ner_pipeline is not None:
        titles = data['Movie Title']
        contents = data['raw_sentences']
        df_dict = {}
        for title, text in zip(titles, contents):
            print ("Processing: ", title)
            entities = ner_pipeline(text)
            df_dict[title] = pd.DataFrame(entities)
        print("NER processing completed.")
        return df_dict
    else:
        print("Data or NER pipeline is not properly initialized.")
        return {}

NERp = setup_ner_pipeline()
aDF = xDF[xDF["Movie Title"] == "The_Toxic_Avenger"]
df_dict = perform_ner_and_store_in_df(aDF, NERp)

# Initialize an empty list to collect all words (entities)
all_entities = []
title = "The_Toxic_Avenger"
# Iterate through each row of the DataFrame
# for title in xDF["Movie Title"].unique():
if title in df_dict:
    df_iter = df_dict[title]
    for index, row in df_iter.iterrows():
        # Iterate through each cell in the row
        for cell in row:
            # Check if the cell contains a dictionary and has the expected keys
            if isinstance(cell, dict) and {'entity_group', 'score', 'word', 'start', 'end'}.issubset(cell):
                # Extract entity information from the dictionary
                entity_group = cell['entity_group']
                score = cell['score']
                word = cell['word']
                start = cell['start']
                end = cell['end']

                # Collect the entity details
                all_entities.append({
                    "Entity": word,
                    "Type": entity_group,
                    "Score": score,
                    "Start": start,
                    "End": end
                })

                # Example: Print out the entity information
                print(f"Entity: {word}, Type: {entity_group}, Score: {score}, Start: {start}, End: {end}")

all_entities = []

# Iterate through each row in the DataFrame
for index, row in df_dict[title].iterrows():
    # Iterate through each column/cell in the row
    for col in row.index:
        # Access the cell value, which is expected to be a dictionary
        cell_value = row[col]
        # Check if the cell value is a dictionary and has the key 'word'
        if isinstance(cell_value, dict) and 'word' in cell_value:
            # Add the "word" value from the dictionary to the list
            all_entities.append(cell_value['word'])

all_entities= [word.lower() for word in all_entities]
entities_kg_df= find_sources_and_targets_with_patterns(all_entities, kg_df)

plot_graph(entities_kg_df, use_circular=True)


## PART 2 ========== LSTM RNN ##
def get_TF_ProbAccuracyScores(NAME, MODEL, X, Y):
    probs = MODEL.predict(X)
    pred_list = []
    for p in probs:
        pred_list.append(np.argmax(p))
    pred = np.array(pred_list)
    acc_score = metrics.accuracy_score(Y, pred)
    return [NAME, acc_score, pred]

def TF_printOverfit(NAME, Train, Test):
    print(NAME, "TRAIN AUC: ", Train[4])
    print(NAME, "TEST AUC: ", Test[4])
    print(NAME, "OVERFIT: ", Train[4] - Test[4])


def print_ROC_Curve(TITLE, LIST):
    fig = plt.figure(figsize=(6, 4))
    plt.title(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        fpr = theResults[2]
        tpr = theResults[3]
        auc = theResults[4]
        theLabel = "AUC " + NAME + ' %0.2f' % auc
        plt.plot(fpr, tpr, label=theLabel)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def print_Accuracy(TITLE, LIST):
    print(TITLE)
    for theResults in LIST:
        NAME = theResults[0]
        ACC = theResults[1]
        print(NAME, " = ", ACC)
    print("------\n")

embSize = 100

random.seed(1212)
xTrain, xTest, yTrain, yTest = train_test_split(xDF, yDF, train_size = .8, test_size = .2, random_state=1)

TrainDocuments = [Document(x, y) for x, y in zip(xTrain.Doc_ID, xTrain.Text)]
TestDocuments = [Document(x, y) for x, y in zip(xTest.Doc_ID, xTest.Text)]

Train_descriptors_by_doc_ids = {x: y for x, y in zip(xTrain['Doc_ID'], xTrain['Descriptor'])}
Test_descriptors_by_doc_ids = {x: y for x, y in zip(xTest['Doc_ID'], xTest['Descriptor'])}

TrainDocuments = clean_method(TrainDocuments)
d2vTrain = run_doc2vec(TrainDocuments, embedding_size = embSize, descriptors_by_doc_ids = Train_descriptors_by_doc_ids)

TestDocuments = clean_method(TestDocuments)
d2vTest = run_doc2vec(TestDocuments, embedding_size = embSize, descriptors_by_doc_ids = Test_descriptors_by_doc_ids)

# w2vTrain = run_doc2vec(TrainDocuments, embedding_size = embSize)
# w2vTest = get_word2vec_vectors(TestDocuments, embedding_size = embSize)

Encoder = LabelEncoder()
TrainE = Encoder.fit_transform(yTrain)
TestE = Encoder.fit_transform(yTest)

WHO = "MDL1"
print("\n\nModel", WHO)
print("1-layer LSTM")

print(d2vTrain[0])
DocCount = len(d2vTrain)

FShape = d2vTrain.shape[1]
print(d2vTrain.shape)
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 300 # iterations
FUnits = int(2*FShape) # nodes
print("Doc Count:", DocCount)
print("Embedding Size:", FShape)
print("Nodes:", FUnits)
print("len(Target Predictions):", len(yTrain))

FLayIn = tf.keras.layers.Embedding(DocCount, FShape)
FLay1 = tf.keras.layers.LSTM(units = FUnits, activation = FActiv)
FLayOut = tf.keras.layers.Dense(units = 4, activation = tf.keras.activations.softmax)

MDL1 = tf.keras.Sequential()
MDL1.add(FLayIn)
MDL1.add(FLay1)
MDL1.add(FLayOut)
MDL1.compile(loss = FLoss, optimizer = FOptim)
MDL1.fit(d2vTrain, TrainE, epochs = FEpoch, verbose = False)

Accs = []
F1TrainM1 = get_TF_ProbAccuracyScores(WHO + "_Train", MDL1, d2vTrain, TrainE)
F1TestM1 = get_TF_ProbAccuracyScores(WHO + "_Test", MDL1, d2vTest, TestE)
Accs.append(F1TestM1)

print_Accuracy(WHO + " F1a CLASSIFICATION ACCURACY", [F1TrainM1, F1TestM1])


# MODEL 2 RNN Classification
Encoder = LabelEncoder()
TrainE = Encoder.fit_transform(yTrain)
TestE = Encoder.fit_transform(yTest)

WHO = "MDL2"
print("\n\nModel", WHO)
print("Bidirectional LSTM")

FShape = d2vTrain.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 300 # iterations
FUnits = int(2*FShape) # nodes
print("Doc Count:", DocCount)
print("Embedding Size:", FShape)
print("Nodes:", FUnits)
print("len(Target Predictions):", len(yTrain))

FLayIn = tf.keras.layers.Embedding(DocCount, FShape)
FLay1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(FShape))
FLayOut = tf.keras.layers.Dense(units = 4, activation = tf.keras.activations.softmax)

MDL2 = tf.keras.Sequential()
MDL2.add(FLayIn)
MDL2.add(FLay1)
MDL2.add(FLayOut)
MDL2.compile(loss = FLoss, optimizer = FOptim)
MDL2.fit(d2vTrain, TrainE, epochs = FEpoch, verbose = False)

F1TrainM2 = get_TF_ProbAccuracyScores(WHO + "_Train", MDL2, d2vTrain, TrainE)
F1TestM2 = get_TF_ProbAccuracyScores(WHO + "_Test", MDL2, d2vTest, TestE)
Accs.append(F1TestM2)

print_Accuracy(WHO + " F2a CLASSIFICATION ACCURACY", [F1TrainM2, F1TestM2])


# MDL 3
WHO = "MDL3"
print("\n\nModel", WHO)
print("2 layer LSTM: 400 iterations")

FShape = d2vTrain.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 400 # iterations
FUnits = int(2*FShape) # nodes
print("Doc Count:", DocCount)
print("Embedding Size:", FShape)
print("Nodes:", FUnits)
print("len(Target Predictions):", len(yTrain))

FLayIn = tf.keras.layers.Embedding(DocCount, FShape)
FLay1 = tf.keras.layers.LSTM(units = FUnits, activation = FActiv)
FLay2 = tf.keras.layers.LSTM(units = FUnits, activation = FActiv)
FLayOut = tf.keras.layers.Dense(units = 4, activation = tf.keras.activations.softmax)

MDL3 = tf.keras.Sequential()
MDL3.add(FLayIn)
MDL3.add(FLay1)
MDL3.add(FLayOut)
MDL3.compile(loss = FLoss, optimizer = FOptim)
MDL3.fit(d2vTrain, TrainE, epochs = FEpoch, verbose = False)

F1TrainM3 = get_TF_ProbAccuracyScores(WHO + "_Train", MDL3, d2vTrain, TrainE)
F1TestM3 = get_TF_ProbAccuracyScores(WHO + "_Test", MDL3, d2vTest, TestE)
Accs.append(F1TestM3)

print_Accuracy(WHO + " F3a CLASSIFICATION ACCURACY", [F1TrainM3, F1TestM3])


# MDL 4
WHO = "MDL4"
print("\n\nModel", WHO)
print("2 Layer LSTM: 1/2 Nodes")

FShape = d2vTrain.shape[1]
FActiv = tf.keras.activations.relu
FLoss = tf.keras.losses.SparseCategoricalCrossentropy()
FOptim = tf.keras.optimizers.Adam()
FEpoch = 300 # iterations
FUnits = FShape # nodes
print("Doc Count:", DocCount)
print("Embedding Size:", FShape)
print("Nodes:", FUnits)
print("len(Target Predictions):", len(yTrain))

FLayIn = tf.keras.layers.Embedding(DocCount, FShape)
FLay1 = tf.keras.layers.LSTM(units = FUnits, activation = FActiv)
FLay2 = tf.keras.layers.LSTM(units = FUnits, activation = FActiv)
FLayOut = tf.keras.layers.Dense(units = 4, activation = tf.keras.activations.softmax)

MDL4 = tf.keras.Sequential()
MDL4.add(FLayIn)
MDL4.add(FLay1)
MDL4.add(FLayOut)
MDL4.compile(loss = FLoss, optimizer = FOptim)
MDL4.fit(d2vTrain, TrainE, epochs = FEpoch, verbose = False)

F1TrainM4 = get_TF_ProbAccuracyScores(WHO + "_Train", MDL4, d2vTrain, TrainE)
F1TestM4 = get_TF_ProbAccuracyScores(WHO + "_Test", MDL4, d2vTest, TestE)
Accs.append(F1TestM4)

print_Accuracy(WHO + " F4a CLASSIFICATION ACCURACY", [F1TrainM4, F1TestM4])


for i in range(0, 4):
    print("Mdl" + str(i+1) + "Acc", Accs[i])