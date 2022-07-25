"""
Description : This file implements the Drain algorithm for log parsing
Author      : LogPAI team
License     : MIT
"""

import regex as re
import os
import numpy as np
import pandas as pd
import hashlib
from datetime import datetime

class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', st=0.4, rex=[], keep_para=True):
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
        self.st = st
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.vectors = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para
        self.clusters = None
        self.cluster_num = None
        self.cluster_labels = None
        self.parsed_sentences = None
        self.word_dict = None

    ## Transforma o dataset
    def transform_dataset(self, raw_content):
        from sentence_transformers import SentenceTransformer
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer('all-mpnet-base-v2')

        self.vectors = model.encode(raw_content)

    def cluster_vectors(self):
        import hdbscan
        import umap

        clusterer = hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1,metric='euclidean',
                                    allow_single_cluster=False,cluster_selection_method='eom')
        reducer = umap.UMAP(n_neighbors=2, n_components=1, spread=0.5, min_dist=0.0, metric='cosine')

        umap_data = reducer.fit_transform(self.vectors)
        self.clusters = clusterer.fit(umap_data)
        self.cluster_num = clusterer.labels_.max()
        self.cluster_labels = clusterer.labels_

    def create_dict_pd(self, df_sentences):
    
        from nltk.tokenize import word_tokenize

        ## Cria dataframe
        values = pd.DataFrame(columns=['Token', 'Cluster', 'Frequence', 'Type'])

        ## Varre os tokens de cada frase
        for id, row in df_sentences.iterrows():
            sentence_tokens = word_tokenize(row.item())
            sentence_cluster = self.cluster_labels[id]
            
            for token in sentence_tokens:
                query = values.query("Token == @token")
                ## Caso o token já exista no dict
                for index, result in query.iterrows():
                    ## Verifica se o cluster é o mesmo da frase. Se for, aumenta frequência
                    if (result['Cluster'] == sentence_cluster):
                        new_frequence = result['Frequence'] + 1
                        values.at[index,'Frequence'] = new_frequence
                        break
                ## Se o token não existir no dict, insere-o no fim
                else:
                    new_val = pd.DataFrame([[token, sentence_cluster, 1, ""]], columns = ['Token', 'Cluster', 'Frequence', 'Type'])
                    values = pd.concat([values,new_val], ignore_index = True)
        
        ### Retorna dicionário de clusters e frequências
        self.word_dict = values    

    ## Carrega os arquivos
    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    ## Preprocessa os arquivos
    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '<*>', line)
        return line
        

    ## Carrega dataframe de logs
    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    ## Limpa os arquivos com expressão regular
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
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    ## Pega lista de parâmetros
    def get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex: return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
            
        ## ALTERAR LINHA ABAIXO
        #template_regex = re.sub(r'\\ +', r'\s+', template_regex)
        template_regex = re.sub(r'\\\s+', '\\\s+', template_regex)

        template_regex = "^" + template_regex.replace("\<\*\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        return parameter_list

    ## Parseador
    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName

        ## CARREGA OS DADOS EM UM DATAFRAME
        self.load_data()
        count = 0
        
        self.transform_dataset(self.df_log["Content"])
        
        ## Sem pre-processamento
        self.create_dict_pd(self.df_log["Content"])





        for idx, line in self.df_log.iterrows():
            #print(idx)
            #print(line)            
            #logID = line['LineId']
            #logmessageL = self.preprocess(line['Content']).strip().split()

            ## CONTEÚDO DO PARSER        
            ## CONTEÚDO DO PARSER        
            ## CONTEÚDO DO PARSER        
            ## CONTEÚDO DO PARSER        
            ## CONTEÚDO DO PARSER

            # Contador de progresso
            count += 1
            if count % 1000 == 0 or count == len(self.df_log):
                print('Processed {0:.1f}% of log lines.'.format(count * 100.0 / len(self.df_log)))


            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

            #self.outputResult(logCluL)

            print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))