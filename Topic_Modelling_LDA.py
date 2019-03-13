#################### 1. READ IN HEADLINES AS A DATAFRAME ######################
import pandas as pd

data = pd.read_csv('abcnews-date-text.csv', error_bad_lines = False)
n_articles = 300000     # only read in the first n_articles headlines
data_text = data[:n_articles][['headline_text']]
data_text['index'] = data_text.index
documents = data_text


########################## 2. PRE-PROCESS HEADLINES ###########################
import gensim
from gensim.utils import simple_preprocess  # lowervase, punctuation removal, tokenization
from gensim.parsing.preprocessing import STOPWORDS # stopwords removal

from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer  # lemmatization
from nltk.stem import SnowballStemmer    # stemming

stemmer = SnowballStemmer('english')

def preprocess(text):
    
    result = []
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            lemmatized = WordNetLemmatizer().lemmatize(token, pos = 'v')
            result.append( SnowballStemmer('english').stem(lemmatized) )
    
    return result


documents['headline_text'] = documents['headline_text'].map(lambda text:preprocess(text))
processed_docs = documents['headline_text'].values


####### 3. CREATE A DICTIONARY OF UNIQUE WORDS ACROSS ALL DOCUMENTS ###########
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below = 15, no_above = 0.1)
print("Number of unique words across ALL documents: \n", len(dictionary))


###### 4. CREATE BAG-OF-WORDS REPRESENTATION OF HEADLINES & UNIQUE WORDS ######
BOW_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]


############ 5. EXECUTE LATENT DRICHLET ALLOCATIONs ON BOW MODEL ##############
from gensim.models import LdaMulticore

BOW_lda_model = LdaMulticore(corpus = BOW_corpus,
                             num_topics = 10,    # we're assuming there are 10 topics
                             id2word = dictionary,
                             workers = 2,
                             chunksize = 2000,
                             passes = 5,
                             batch = False,
                             alpha = 'symmetric', # LDA params for doc-topics (all = 0.1)
                             eta = None)          # LDA params for topic-words (all = 0.1)
'''
: param corpus     : corpus to perform the LDA on
: param num_topics : assumed number of topics present in the corpus
: param id2word    : dictionary mapping word ids (int) to actual words (str)
: param alpha      : list of parameters for Drichlet Distribution of topics per document
                    --> # of parameters = num_topics  (# of topics)
                    --> if 'symmetric', all parameters = 0.1
: param eta        : list of parameters for Drichlet Distribution of words per topic
                    --> # of parameters = len(id2word) (# of unique words)      
                    --> if not specified, all parameters = 0.1
'''

############### 6. PRINT OUT DETECTED TOPICS & ASSOCIATED WORDS ###############
# Following prints out words occuring in each of the 10 topics & their relative weight
for i, topic in BOW_lda_model.print_topics(-1):
    print("Topic {}: \n{}\n".format(i, topic))


############### 7. PREDICT A TOPIC CLASS FOR A SAMPLE DOCUMENT ################
# Use BOW_lda_model to predict which topic this document belongs to:
sample_doc_i = 827

for i, score in sorted(BOW_lda_model[BOW_corpus[sample_doc_i]], key = lambda tup:-1*tup[1]):
    print("\nScore: {}\nTopic: {}".format(score, BOW_lda_model.print_topic(i, 10)))


################# 8. PREDICT A TOPIC CLASS FOR A NEW DOCUMENT #################
# Use BOW_lda_model to predict which topic a new document belongs to:
new_doc = "Syria gets terrorist attack kills 22 people"

new_BOW_vector = dictionary.doc2bow( preprocess(new_doc) )

for i, score in sorted(BOW_lda_model[new_BOW_vector], key = lambda tup:-1*tup[1]):
    print("Score: {}\nTopic: {}\n".format(score, BOW_lda_model.print_topic(i, 10)))


######### 9. CREATE TF-IDF REPRESENTATION OF HEADLINES & UNIQUE WORDS #########
from gensim.models import TfidfModel

TFIDF_model = TfidfModel(BOW_corpus)
TFIDF_corpus = [TFIDF_model[doc] for doc in BOW_corpus]


########### 10. EXECUTE LATENT DRICHLET ALLOCATION ON TF-IDF MODEL ############
TFIDF_lda_model = LdaMulticore(corpus = BOW_corpus,
                               num_topics = 10,
                               id2word = dictionary,
                               workers = 2,
                               chunksize = 2000,
                               passes = 5,
                               batch = False,
                               alpha = 'symmetric',
                               eta = None)


############### 11. PRINT OUT DETECTED TOPICS & ASSOCIATED WORDS ##############
# Following prints out words occuring in each of the 10 topics & their relative weight
for i, topic in TFIDF_lda_model.print_topics(-1):
    print("Topic {}: \n{}\n".format(i, topic))


############## 12. PREDICT A TOPIC CLASS FOR A SAMPLE DOCUMENT ################
# Use TFIDF_lda_model to predict which topic this document belongs to:
sample_doc_i = 827

for i, score in sorted(BOW_lda_model[TFIDF_corpus[sample_doc_i]], key = lambda tup:-1*tup[1]):
    print("\nScore: {}\nTopic: {}".format(score, BOW_lda_model.print_topic(i, 10)))