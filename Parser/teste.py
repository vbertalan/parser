import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
import py_stringmatching as sm

alnum_tok = sm.AlphanumericTokenizer()
#result = alnum_tok.tokenize('data9,(science), data9#.(integration).88')

#print(result)



frase = "logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root"
print(alnum_tok.tokenize(frase))

# def my_tokenizer(sentence):
#     new_sentence = []
#     for char in sentence:
#         word = ""


#         new_sentence.append(new_word)

#teste = "ALOOOO"
#print(teste.lower())





