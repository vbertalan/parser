"""
Description : This file implements the my algorithm for log parsing
Author      : Vithor Bertalan
License     : n/a
"""

import contextlib
from fileinput import filename
import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import py_stringmatching as sm
import pickle

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', vecdir='./', rex=None, threshold = 0, filename=""):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            st : similarity threshold
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.vectors = None
        self.log_format = log_format
        self.rex = rex
        self.clusters = None
        self.cluster_num = None
        self.cluster_labels = None
        self.parsed_sentences = None
        self.word_dict = None
        self.threshold = threshold
        self.outdir = outdir
        self.vecdir = vecdir
        self.filename = filename

    ## Transforma o dataset
    def transform_dataset(self, raw_content):
        
        from pathlib import Path
        path_to_file = os.path.join(self.vecdir, self.filename + '_vectors.vec')
        path = Path(path_to_file)

        if (path.is_file()):
            self.vectors = pickle.load(open(path_to_file, 'rb'))
        else:
            from sentence_transformers import SentenceTransformer
            #model = SentenceTransformer('all-MiniLM-L6-v2')
            model = SentenceTransformer('all-mpnet-base-v2')
            #model = SentenceTransformer('/home/vberta/projects/def-aloise/vberta/Parser-CC/parser/Ciena-Full-Transformer')
            vectors = model.encode(raw_content)
            self.vectors = vectors
            pickle.dump(vectors, open(path_to_file, 'wb'))

    def new_tokenizer(self, sentence):
        new_sentence = []
        new_word = ""
        for char in sentence:
            if char not in [" ", "="]:
                new_word += char
            elif len(new_word) > 0:
                new_sentence.append(new_word)
                new_word = ""
        new_sentence.append(new_word)
        return new_sentence

    def cluster_vectors(self):
        import hdbscan
        import umap
        
        clusterer = hdbscan.HDBSCAN(min_cluster_size=1,min_samples=1,metric='euclidean',
                                    allow_single_cluster=False,cluster_selection_method='leaf')
        #reducer = umap.UMAP(n_neighbors=2, n_components=1, spread=0.5, min_dist=0.0, metric='cosine')

        #umap_data = reducer.fit_transform(self.vectors)
        #self.clusters = clusterer.fit(umap_data)

        self.cluster = clusterer.fit(self.vectors)
        self.cluster_num = clusterer.labels_.max()
        self.cluster_labels = clusterer.labels_

    def cluster_vector_agglomerative(self):
        from sklearn.cluster import AgglomerativeClustering
        sk_clusterer = AgglomerativeClustering().fit(self.vectors)
        self.cluster_labels = sk_clusterer.labels_

    def cluster_vector_kmeans(self):
        from sklearn.cluster import KMeans
        ## Clusteriza com K-Means
        num_clusters = 30
        clustering_model = KMeans(n_clusters=num_clusters)
        clustering_model.fit(self.vectors)
        self.cluster_labels = clustering_model.labels_
    

    def create_dict(self, df_sentences):        

        values = pd.DataFrame(columns=['Token', 'Cluster', 'Frequence', 'Type'])
        tokenizer = sm.AlphanumericTokenizer()

        for id, row in tqdm(df_sentences.iteritems(), desc="Creating Dict", total=len(df_sentences)):
            #sentence_tokens = word_tokenize(row)
            lowercase_row = row.lower()
            sentence_tokens = self.new_tokenizer(lowercase_row)
            #sentence_tokens = tokenizer.tokenize(lowercase_row)
            sentence_cluster = self.cluster_labels[id]
            
            for token in sentence_tokens:
                query = values.query("Token == @token")
                for index, result in query.iterrows():
                    if (result['Cluster'] == sentence_cluster):
                        new_frequence = result['Frequence'] + 1
                        values.at[index,'Frequence'] = new_frequence
                        break
                else:
                    new_val = pd.DataFrame([[token, sentence_cluster, 1, ""]], columns = ['Token', 'Cluster', 'Frequence', 'Type'])
                    values = pd.concat([values,new_val], ignore_index = True)

        self.word_dict = values    

    def set_types(self, token_dict, cluster_labels, percentage):

        for label in tqdm(cluster_labels, desc="Setting Types", total=len(cluster_labels)):
            
            query = token_dict.query("Cluster == @label")
            max_frequence = query['Frequence'].max()

            for index, result in query.iterrows():
                current_frequence = result['Frequence']
                current_threshold = current_frequence / max_frequence
                if (current_threshold < percentage):
                    token_dict.at[index,'Type'] = "VARIABLE"
                else:
                    token_dict.at[index,'Type'] = "STATIC"
        
        self.word_dict = token_dict

    ## Method to split words and verify if they are in the English language
    # def split_words(self, sentence):

    #     with open("create.txt") as word_file:
    #         english_words = {word.strip().lower() for word in word_file}

    #         tokenized_words = word_tokenize(sentence)
    #         new_sentence = []

    #         for tokens in tokenized_words:
    #             splits = wordninja.split(tokens)
    #             for word in splits:
    #                 if word.lower() not in english_words:
    #                     new_sentence.append("<*>")
    #                     break
    #             else:
    #                 new_sentence.append(tokens)

    #         return new_sentence

    ## Method to check if a word is present in the English language, without splitting the tokens
    # def check_english(self, sentence):

    #     with open("create.txt") as word_file:
    #         english_words = {word.strip().lower() for word in word_file}

    #         tokenized_words = word_tokenize(sentence)
    #         new_sentence = []
    #         for token in tokenized_words:
    #             if token.lower() not in english_words:
    #                 new_sentence.append("<*>")
    #             else:
    #                 new_sentence.append(token)       

    #     return new_sentence

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def preprocess_df(self):
        for idx, content in self.df_log["Content"].iteritems():
            for currentRex in self.rex:
                self.df_log.at[idx,'Content'] = re.sub(currentRex, '<*>', content)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                with contextlib.suppress(Exception):
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName

        self.load_data()
        self.preprocess_df()
        self.transform_dataset(self.df_log["Content"])
        #self.cluster_vectors()
        #self.cluster_vector_agglomerative()
        self.cluster_vector_kmeans()
        self.create_dict(self.df_log["Content"])
        self.set_types(self.word_dict, self.cluster_labels, self.threshold)

        self.word_dict.to_csv('out.csv', index=False)

        log_templates = []
        log_templateids = []

        tokenizer = sm.AlphanumericTokenizer()

        create_path = "C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser-1\\create.txt"
        #create_path = "/home/vbertalan/Downloads/Parser/parser/create.txt"

        with open(create_path) as word_file:
            english_words = {word.strip().lower() for word in word_file}

        print('\n=== Parsing dataset ===')
        for count, (idx, line) in tqdm(enumerate(self.df_log.iterrows(), start=1), desc="Parsing Lines", total=len(self.df_log)):
            sentence_cluster = self.cluster_labels[idx]
            #sentence_tokens = word_tokenize(line["Content"])
            #sentence_tokens = tokenizer.tokenize(line["Content"])
            #print(type(sentence_tokens))
            #print(sentence_tokens)
            #print("************")
            sentence_tokens = self.new_tokenizer(line["Content"]) #LINHA A CORRIGIR
            #print(type(sentence_tokens))
            #print(sentence_tokens)
            #sentence_tokens = self.check_english(line["Content"])
            new_sentence = ""

            for token in sentence_tokens:
                lowercase_token = token.lower()
                #print(lowercase_token)
                query = self.word_dict.query("Cluster == @sentence_cluster & Token == @lowercase_token")
                #print(query)
                ## Checking dictionary of variables only
                #new_token = "<*>" if (query["Type"].item() == 'VARIABLE') else token            
                
                ## Checking English vocabulary only
                #new_token = "<*>" if (token.lower() not in english_words) else token
                
                ## Checking variables or English tokens
                new_token = "<*>" if (query["Type"].item() == 'VARIABLE' or token.lower() not in english_words) else token                           
                
                new_sentence += new_token
                new_sentence += " "

            log_templates.append(new_sentence)
            log_templateids.append(hashlib.md5(new_sentence.encode('utf-8')).hexdigest()[:8])


            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))

            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

            #print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

        #print(len(self.word_dict))

        self.df_log['EventId'] = log_templateids
        self.df_log['EventTemplate'] = log_templates

        self.df_log.to_csv(os.path.join(self.savePath, self.logName + '_structured.csv'), index=False)

        occ_dict = dict(self.df_log['EventTemplate'].value_counts())
        df_event = pd.DataFrame()
        df_event['EventTemplate'] = self.df_log['EventTemplate'].unique()
        df_event['EventId'] = df_event['EventTemplate'].map(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[:8])

        df_event['Occurrences'] = df_event['EventTemplate'].map(occ_dict)
        df_event.to_csv(os.path.join(self.savePath, self.logName + '_templates.csv'), index=False, columns=["EventId", "EventTemplate", "Occurrences"])
