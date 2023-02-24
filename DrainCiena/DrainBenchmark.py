#!/usr/bin/env python

import sys
# Path - Windows
#sys.path.append("C:/Users/vbert/OneDrive/DOUTORADO Poly Mtl/Projeto/pyteste")
# Path - Linux
sys.path.append("/home/vbertalan/Downloads/Parser/parser/DrainCiena")
import evaluator
import DrainCiena
import os
import pandas as pd
from pathlib import Path

input_dir = "DrainCiena/logs/"
output_dir = "DrainCiena/result/"  # The output directory of parsing results

benchmark_settings = {
    'HDFS': {
        # Name of the log file
        'log_file': 'HDFS_2k.log',

        # With pre-defined log formats and structures of the lines
        #'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',

        # Without pre-defined log formats
        'log_format': '<Content>',

        # In case we know the regex of the lines
        #'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],

        # If we do not know the regex of the lines
        'regex': [],

        # Similarity threshold
        'st': 0.5,

        # Max depth of the parsing tree
        'depth': 4
        },
}

bechmark_result = []
for dataset, setting in benchmark_settings.items():
    print('\n=== Evaluation on %s ==='%dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    parser = DrainCiena.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'], depth=setting['depth'], st=setting['st'])
    parser.parse(log_file)
    
    F1_measure, accuracy = evaluator.evaluate(
                           groundtruth=os.path.join(indir, log_file + '_structured.csv'),
                           parsedresult=os.path.join(output_dir, log_file + '_structured.csv')
                           )
    bechmark_result.append([dataset, F1_measure, accuracy])


print('\n=== Overall evaluation results ===')
df_result = pd.DataFrame(bechmark_result, columns=['Dataset', 'F1_measure', 'Accuracy'])
df_result.set_index('Dataset', inplace=True)
print(df_result)
filepath = Path('Drain_bechmark_result.csv') 
#df_result.T.to_csv('Drain_bechmark_result.csv')
df_result.T.to_csv(filepath)

