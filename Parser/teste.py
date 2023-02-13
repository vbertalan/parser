import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.utils import tokenize
#import py_stringmatching as sm

#alnum_tok = sm.AlphanumericTokenizer()
#result = alnum_tok.tokenize('data9,(science), data9#.(integration).88')
#print(result)

frase = "authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net  user=root"
#print(alnum_tok.tokenize(frase))

def my_tokenizer(sentence):
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

#print(my_tokenizer(frase))

path = "C:\\Users\\vbert\\OneDrive\\DOUTORADO Poly Mtl\\Projeto\\parser\\logparser\\logs\\Linux\\Linux_2k.log_structured.csv"
path_teste = "C:\\Users\\vbert\\Downloads\\Linux_2k.log_structured - Mini.csv"
cluster_labels = ""

def conta_tokens (new_file): 
    df = pd.read_csv(new_file)
    df = df["Content"]
    values = pd.DataFrame(columns=['Token', 'Freq'])
    i = 0

    for line in df:
        tokenized_line = my_tokenizer(line)
        
        print("parseando linha {}".format(i))
        for token in tokenized_line:            
            query = values.query("Token == @token")
            for index, result in query.iterrows():
                new_frequence = result['Freq'] + 1
                values.at[index,'Freq'] = new_frequence
                break
            else:
                new_val = pd.DataFrame([[token, 1]], columns = ['Token', 'Freq'])
                values = pd.concat([values,new_val], ignore_index = True)  
        i += 1
    values.sort_values(by='Freq', ascending=False)
    values.to_csv('C:\\Users\\vbert\\Downloads\\result.csv', index=True)  
    #return (values)

def transform_dataset(raw_content):

    df = pd.read_csv(raw_content)
    df = df["Content"]
    from sentence_transformers import SentenceTransformer
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformer('all-mpnet-base-v2')
    vectors = model.encode(df)
    return (vectors)

def cluster_vectors(vectors):
    import hdbscan
    import umap
        
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2,min_samples=1,metric='euclidean',
                                    allow_single_cluster=False,cluster_selection_method='leaf')
    clusterer.fit(vectors)
    cluster_labels = clusterer.labels_
    return (cluster_labels)

def cluster_aglomerativo(vectors):
    from sklearn.cluster import AgglomerativeClustering
    sk_clusterer = AgglomerativeClustering().fit(vectors)
    return(sk_clusterer.labels_)

def cluster_kmeans(vectors):
    from sklearn.cluster import KMeans
    ## Clusteriza com K-Means
    num_clusters = 4
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(vectors)
    return(clustering_model.labels_)
    #cluster_assignment = clustering_model.labels_

    #clustered_sentences = [[] for _ in range(num_clusters)]
    #for sentence_id, cluster_id in enumerate(cluster_assignment):
    #    clustered_sentences[cluster_id].append(sentences[sentence_id])

    #for i, cluster in enumerate(clustered_sentences):
    #    print("Cluster ", i+1)
    #    print(cluster)
    #    print("")

#conta_tokens(path_teste)
dataset_transformado = transform_dataset(path_teste)
clusters_gerados = cluster_aglomerativo(dataset_transformado)
print("Clusters via aglomerativo")
print(clusters_gerados)
clusters_gerados = cluster_kmeans(dataset_transformado)
print("Clusters via kMeans")
print(clusters_gerados)
#clusters_gerados = cluster_vectors(dataset_transformado)
#print(clusters_gerados)




