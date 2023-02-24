#!/usr/bin/env python

from pathlib import Path
import DrainCiena
import sys
import os

# Path - Windows
#sys.path.append("C:/Users/vbert/OneDrive/DOUTORADO Poly Mtl/Projeto/pyteste")
# Path - Linux
sys.path.append("/home/vbertalan/Downloads/Parser/parser/DrainCiena")

input_dir = "DrainCiena/logs/"
output_dir = "DrainCiena/result/"  # The output directory of parsing results

log_datafiles = {
    # 'HDFS': {
    #     # Name of the log file
    #     'log_file': 'HDFS_2k.log',

    #     # With pre-defined log formats and structures of the lines
    #     #'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',

    #     # Without pre-defined log formats
    #     'log_format': '<Content>',

    #     # In case we know the regex of the lines
    #     #'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],

    #     # If we do not know the regex of the lines
    #     'regex': [],

    #     # Similarity threshold
    #     'st': 0.5,

    #     # Max depth of the parsing tree
    #     'depth': 4
    #     },

    'Mac': {
        # Name of the log file
        'log_file': 'Mac.log',

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

# For each of the log datasets, parse the file, and create two resulting files:
# 1 - file_structured.csv, representing the parsed lines
# 2 - file_templates.csv, representing the unique templates found after the parsing

for dataset, setting in log_datafiles.items():
    print('\n=== Evaluation on %s ==='%dataset)
    indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
    log_file = os.path.basename(setting['log_file'])

    parser = DrainCiena.LogParser(log_format=setting['log_format'], indir=indir, outdir=output_dir, rex=setting['regex'], depth=setting['depth'], st=setting['st'])
    parser.parse(log_file)

    parsedresult=os.path.join(output_dir, log_file + '_structured.csv')   
    
print('\n=== Parsing finished ===')