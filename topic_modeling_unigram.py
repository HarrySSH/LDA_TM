
### Now this code can remove the word that is shorter than 3 letters


import nltk
import smart_open

smart_open.open = smart_open.smart_open
import re
import numpy as np
import pandas as pd
from pprint import pprint
import os
import argparse

parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('--num', default=False, type=int)
parser.add_argument('--time', default=False, type=int)

args = parser.parse_args()
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this
import matplotlib.pyplot as plt
# %matplotlib inline

# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# NLTK Stop words
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use',
                   'social', 'pt', 'note', 'pm', 'am', 'sw',
                   'time','need','min','year','date','form','role',
                   'also','likely','work','center','week',
                   'group','peer','self','dbt', 'month',
                   'si','visit','PT', 'SW'])

# Import Dataset
df = pd.read_csv('/wynton/protected/project/outcome_pred/Harry_Social_Notes/data/social_notes.csv', index_col=0)
meta = pd.read_csv('/wynton/protected/project/outcome_pred/Harry_Social_Notes/data/social_notes_meta.csv', index_col=0)

#df = df.head(1000)
'''
Let’s tokenize each sentence into a list of words, removing punctuations and unnecessary characters altogether.
'''
# Convert to list
data = df.note_text.values.tolist()

data = [str(x) for x in data]
pprint(data[:1])

# Remove new line characters
data = [re.sub('\s+', ' ', sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\a", "", sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\t", "", sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\'", "", sent) for sent in data]

# Remove distracting single quotes
data = [re.sub("\n", "", sent) for sent in data]
pprint(data[:1])


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


data_words = list(sent_to_words(data))

'''
Bigrams are two words frequently occurring together in the document. Trigrams are 3 words frequently occurring.
Some examples in our example are: ‘front_bumper’, ‘oil_leak’, ‘maryland_college_park’ etc.
Gensim’s Phrases model can build and implement the bigrams, trigrams, quadgrams and more. 
The two important arguments to Phrases are min_count and threshold. 
The higher the values of these param, the harder it is for words to be combined to bigrams.
'''
# This is optional, I did not use bigram and trigram for this projects
# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)



# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def remove_shortwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if len(word) > 3] for doc in texts]



def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out







# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
data_words_noshorts = remove_shortwords(data_words)
# Form Trigrams
#data_words_trigrams = make_trigrams(data_words_noshorts)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm')

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_noshorts, allowed_postags=['NOUN', 'ADJ', 'ADV'])


def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))


# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

num_keywords = 10
# Build LDA model LdaModel


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=args.num,

                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

shown_topics = lda_model.show_topics(num_topics=args.num,
                                     num_words=num_keywords,
                                     formatted=False)
lda_topics = [[word[0] for word in topic[1]] for topic in shown_topics]

lda_stability = 0

jaccard_sims = []
print(len(lda_topics))
num = len(lda_topics)
for t1, topic1 in enumerate(lda_topics):  # pylint: disable=unused-variable
    sims = []
    for t2, topic2 in enumerate(lda_topics):  # pylint: disable=unused-variable
        if t1 != t2:
            sims.append(jaccard_similarity(topic1, topic2))

            jaccard_sims.append(sims)
        else:
            pass

    lda_stability = jaccard_sims

mean_stabilities = np.array(lda_stability).mean()

cm = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence = cm.get_coherence()
writepath = '/wynton/protected/project/outcome_pred/Harry_Social_Notes/Topics_unigram_decide_clusters.tsv'
'''
mode = 'a' if os.path.exists(writepath) else 'w'
with open(writepath, mode) as f:
    if mode == 'w':
        f.write('Num_topics' + '\t' + 'Average Topic Overlap' + '\t' + 'Topic Coherence' + '\n')
        f.write(str(args.num) + '\t' + str(mean_stabilities) + '\t' + str(coherence) + '\n')
    else:
        assert mode == 'a', 'something is wrong'
        f.write(str(args.num) + '\t' + str(mean_stabilities) + '\t' + str(coherence) + '\n')
'''
txt_dir = '/wynton/protected/project/outcome_pred/Harry_Social_Notes/diff_cluster/Topics_unigram_' + str(
    args.num) +'time'+str(args.time)+ '.txt'

html_dir = '/wynton/protected/project/outcome_pred/Harry_Social_Notes/diff_cluster/lda_vis_unigram_' + str(
    args.num) +'time'+str(args.time)+'.html' #lda_vis_unigram_


# begin saving the results
#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, html_dir)




word_results = pd.DataFrame(columns=list(range(1, 11)))
for i in range(len(lda_model.print_topics())):
    print(i)
    word_list = []

    for word in lda_model.print_topics()[i][1].split('*')[1:]:
        word = word.split('"')[1]

        word_list.append(word)
    add_word_results = pd.DataFrame(np.array([word_list]), columns=list(range(1, 11)))
    word_results = word_results.append(add_word_results).copy()

word_results['Freq'] = ''
word_results.index = list(range(word_results.shape[0]))
for i in range(len(lda_model.print_topics())):
    word_results['Freq'][i] =vis[0]['Freq'][i]
word_results.to_csv(txt_dir, sep ='\t')