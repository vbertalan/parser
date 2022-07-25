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
    def __init__(self, log_format, indir='./', outdir='./result/', depth=4, st=0.4, 
                 maxChild=100, rex=[], keep_para=True):
        """
        Attributes
        ----------
            rex : regular expressions used in preprocessing (step1)
            path : the input path stores the input log file name
            depth : depth of all leaf nodes
            st : similarity threshold
            maxChild : max number of children of an internal node
            logName : the name of the input file containing raw log messages
            savePath : the output path stores the file containing structured logs
        """
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.keep_para = keep_para

    ## Transforma o dataset
    def transform_dataset(raw_logs):
        ## First step - import SentenceTransformers
        from sentence_transformers import SentenceTransformer
        #model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer('all-mpnet-base-v2')

        ## Junta as frases
        #sentences = [log1, log2, log3, log4, log5, log6]

        ## Faz encode nas frases
        embeddings = model.encode(raw_logs)

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

        print(self.df_log)

        count = 0
        for idx, line in self.df_log.iterrows():
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