import os
import glob
import csv
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# a script to aggregate our transcription data in one big csv
# deployed by aussie frost on 05/24/2023

# make imports of preprocessing funcs 
stemmer = PorterStemmer()
stemmer.stem("running")

stop_words = set(stopwords.words('english'))
'do' in stop_words, 'when' in stop_words

def tokenizer(sentence):
    # remove punctuation from sentence
    sentence = ''.join(
        char for char in sentence if char not in string.punctuation
    )
    # tokenizing the sentence
    tokens = nltk.word_tokenize(sentence)
    return [token.lower() for token in tokens]

def stopword_destroyer(tokens): 
  tokens_no_sw = []

  for word in tokens:
    if word not in stop_words:
      tokens_no_sw.append(word)

  return tokens_no_sw

def stemmerizer(tokens):
  stemmed_tokens = []
  for i in tokens:
    stemmed_tokens.append(stemmer.stem(i))
  return stemmed_tokens

def preprocess(sentence):
    """
    This function takes a sentence as input and performs various text preprocessing steps on it,
    including removing punctuation, stop words, and stemming each word in the sentence.
    """
    # tokenizing the sentence
    tokens = tokenizer(sentence)
    
    # removing stop words
    tokens = stopword_destroyer(tokens)

    # stemming each word in the sentence
    tokens = stemmerizer(tokens)

    # return the preprocessed sentence as a list of words
    return tokens

# define paths
working_dir = '/Users/austinfroste/Documents/CSNL/narrative_sentiment_analysis'
data_dir = working_dir + '/Transcripts/Transcripts'
output_dir = working_dir + '/Transcripts/nsa_data'

# for all subjects, get path
sub_dirs = glob.glob('%s/d[0-9][0-9]_[0-9][0-9]'%(data_dir))
sub_dirs = np.array(sub_dirs)
sub_dirs.sort()

# for each sentence, place into csv with all respective info
output_file_name = ('/transcript_aggregate.csv')
output_file = output_dir + output_file_name
os.system('touch %s'%(output_file))
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    field = ['sub_num', 'run_type', 'text', 'sentiment']
    writer.writerow(field)

    for dir in sub_dirs:
        # get sub_num as id
        sub_num = re.findall('d[0-9][0-9]_[0-9][0-9]', dir)[0]
        print('parcing -> ' + sub_num)

        # for all subjects, get file paths to parce
        sub_files = glob.glob('%s/*.txt'%(dir))
        sub_files = np.array(sub_files)
        sub_files.sort()

        # for each file, read into csv sentence by sentence
        for file in sub_files:
            # get run_type and sentiment
            run_type = ''
            sentiment = ''
            if 'pos_past' in file:
                run_type = 'pos_past'
                sentiment = 'positive'
            elif 'pos_fut' in file:
                run_type = 'pos_fut'
                sentiment = 'positive'
            elif 'neg_past' in file:
                run_type = 'neg_past'
                sentiment = 'negative'
            elif 'neg_fut' in file:
                run_type = 'neg_fut'
                sentiment = 'negative'
            elif 'idea' in file:
                run_type = 'idea'
                sentiment = 'neutral'

            # read in file
            text = ''
            with open(file,'r') as f:
                text = f.read()
                f.close()
            
            # parce through text and split sentence by sentence
            sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

            # clean sentences up: arbitrary info
            sentences[0] = sentences[0].split('\n')[1].lstrip() # removes speaker info
            sentences = np.delete(sentences, -2) # removes new line
            sentences = np.delete(sentences, -1) # removes credit line

            # preprocess sentences (do later instead)
            '''
            prep_sentences = []
            for sentence in sentences:
                prep_sentences = np.append(prep_sentences, preprocess(sentence))
            print (prep_sentences)
            '''
            row = []
            for sentence in sentences:
                row = [sub_num, run_type, sentence, sentiment]
                writer.writerow(row)
            

