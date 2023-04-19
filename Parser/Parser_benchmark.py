#!/usr/bin/env python

import sys
# Path - Windows
#sys.path.append("C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser-1\\Parser")
# Path - Linux
#sys.path.append("/home/vbertalan/Downloads/Parser/parser")
from fileinput import filename
import evaluator
import Parser
import os
import pandas as pd
from pathlib import Path

#input_dir = "/home/vbertalan/Downloads/Parser/parser/Parser/logs" # The directory to get the logs
#output_dir = "/home/vbertalan/Downloads/Parser/parser/Parser/results"  # The output directory of parsing results
#vector_dir = "/home/vbertalan/Downloads/Parser/parser/Parser/vectors" # The directory to save the vectorized files

input_dir = "C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser-1\\Parser\\logs" # The directory to get the logs
output_dir = "C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser-1\\Parser\\results"  # The output directory of parsing results
vector_dir = "C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser-1\\Parser\\vectors" # The directory to save the vectorized files

# Dictionary to load files
benchmark_settings = {
      'Ciena-mini': {
        'log_file': 'Ciena/ciena-mini.txt',
        'log_format': '<Content>', 
        'regex': [],
        'threshold': 0.1,
        'accuracy': 0     
        },   
   
    #   'Ciena-full': {
    #     'log_file': 'Ciena/ciena-full.txt',
    #     'log_format': '<Content>', 
    #     'regex': [],
    #     'threshold': 0.1      
    #     },   

    # 'Hadoop': {
    #     'log_file': 'Hadoop/Hadoop_2k.log',
    #     'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'threshold': 0.1      
    #     },

    # 'HDFS': {
    #     'log_file': 'HDFS/HDFS_2k.log',
    #     'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
    #     'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
    #     #'regex': [],
    #     },

    # 'Spark': {
    #     'log_file': 'Spark/Spark_2k.log',
    #     'log_format': '<Date> <Time> <Level> <Component>: <Content>', 
    #     'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Zookeeper': {
    #     'log_file': 'Zookeeper/Zookeeper_2k.log',
    #     'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
    #     'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'BGL': {
    #     'log_file': 'BGL/BGL_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
    #     'regex': [r'core\.\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'HPC': {
    #     'log_file': 'HPC/HPC_2k.log',
    #     'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
    #     'regex': [r'=\d+'],
    #     'st': 0.5,
    #     'depth': 4
    #     },

    # 'Thunderbird': {
    #     'log_file': 'Thunderbird/Thunderbird_2k.log',
    #     'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Windows': {
    #     'log_file': 'Windows/Windows_2k.log',
    #     'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
    #     'regex': [r'0x.*?\s'],
    #     'st': 0.7,
    #     'depth': 5      
    #     },

    # 'Linux': {
    #     'log_file': 'Linux/Linux_2k.log',
    #     'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
    #     'st': 0.39,
    #     'depth': 6        
    #     },

    # 'Andriod': {
    #     'log_file': 'Andriod/Andriod_2k.log',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'st': 0.2,
    #     'depth': 6   
    #     },

    # 'HealthApp': {
    #     'log_file': 'HealthApp/HealthApp_2k.log',
    #     'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
    #     'regex': [],
    #     'st': 0.2,
    #     'depth': 4
    #     },

    # 'Apache': {
    #     'log_file': 'Apache/Apache_2k.log',
    #     'log_format': '\[<Time>\] \[<Level>\] <Content>',
    #     'regex': [r'(\d+\.){3}\d+'],
    #     'st': 0.5,
    #     'depth': 4        
    #     },

    # 'Proxifier': {
    #     'log_file': 'Proxifier/Proxifier_2k.log',
    #     'log_format': '\[<Time>\] <Program> - <Content>',
    #     'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
    #     'st': 0.6,
    #     'depth': 3
    #     },

    # 'OpenSSH': {
    #     'log_file': 'OpenSSH/OpenSSH_2k.log',
    #     'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
    #     'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.6,
    #     'depth': 5   
    #     },

    # 'OpenStack': {
    #     'log_file': 'OpenStack/OpenStack_2k.log',
    #     'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
    #     'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
    #     'st': 0.5,
    #     'depth': 5
    #     },

    # 'Mac': {
    #     'log_file': 'Mac/Mac_2k.log',
    #     'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
    #     'regex': [r'([\w-]+\.){2,}[\w-]+'],
    #     'st': 0.7,
    #     'depth': 6   
    #     },

}

benchmark_result = []
test_accuracy = False

## Single threshold
for dataset, setting in benchmark_settings.items():
    print('\n=== Evaluation on %s ==='%dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    parser = Parser.LogParser(log_format=setting['log_format'], indir=indir, 
                                outdir=output_dir, vecdir=vector_dir, rex=setting['regex'], threshold = setting['threshold'], filename=log_file)
    parser.parse(log_file)  

    if (setting['accuracy'] == 0):
        ## REMOVER LINHA SE TESTANDO ACURÁCIA
        parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
        test_accuracy = True
        print('\n=== Parsing finished ===')
    else:
        F1_measure, accuracy = evaluator.evaluate(
                            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                            )
        benchmark_result.append([dataset, F1_measure, accuracy])


    
## Testing different thresholds
# for dataset, setting in benchmark_settings.items():
#     print('\n=== Evaluation on %s ==='%dataset)
#     indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
#     log_file = os.path.basename(setting['log_file'])
#     threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]
#     best_F1_measure = 0
#     best_accuracy = 0
#     best_threshold = 0

#     empty_array = []
#     for number in threshold_list:
#         parser = Parser.LogParser(log_format=setting['log_format'], indir=indir, 
#                                 outdir=output_dir, vecdir=vector_dir, rex=empty_array, threshold = number, filename=log_file)
#         parser.parse(log_file)    

#         F1_measure, accuracy = evaluator.evaluate(
#                            groundtruth=os.path.join(indir, log_file + '_structured.csv'),
#                            parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
#                            )
        
#         if F1_measure > best_F1_measure:
#             best_F1_measure = F1_measure
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy    
#             best_threshold = number
    
#     print("On dataset {}, the best threshold is {}!".format(dataset, best_threshold))
#     benchmark_result.append([dataset, best_F1_measure, best_accuracy])

if (test_accuracy):
    print('\n=== Overall evaluation results ===')
    df_result = pd.DataFrame(benchmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
    df_result.set_index('Dataset', inplace=True)
    print(df_result)
    path_to_file = os.path.join(output_dir, 'Parser_benchmark_result.csv')
    filepath = Path(path_to_file)
    df_result.T.to_csv(filepath)

