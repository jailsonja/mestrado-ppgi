from model import *
from utils import *
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ast
from preprocess import *
import numpy as np
import os

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

gabs = ["../data/gabarito/DadosCamera.txt", "../data/gabarito/DadosCells.txt", '../data/gabarito/DadosDvds.txt', '../data/gabarito/DadosLaptops.txt', '../data/gabarito/DadosRouters.txt']

dict_product = {'camera': 0, 'cells': 1, 'dvds': 2, 'laptops': 3, 'routers': 4}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', default='/contextual_topic_identification/data/steam_reviews.csv')
    parser.add_argument('--ntopic', default=10)
    parser.add_argument('--method', default='TFIDF')
    parser.add_argument('--samp_size', default=10000)
    args = parser.parse_args()
    
    gabarito = set()
    # Obtendo o gabarito vindo de um arquivo
    arq = open(gabs[4],"r")
    for linha in arq.readlines():
        valores = linha.split(":")
        v = ast.literal_eval(valores[1])
        gabarito = gabarito.union(v)
    arq.close()
    
    #Tranformando o gabarito em texto/sentenca
    gabarito = " ".join(list(gabarito))
    
    #Obtendo o gabarito preprocessado em forma de lista de tokens
    gabarito_preprocess, not_steming = preprocess_word_alt(gabarito)
    gabarito_preprocess = set(gabarito_preprocess)
    not_steming = set(not_steming)
    
    data = pd.read_csv(str(args.fpath))
    data = data.fillna('')  # only the comments has NaN's
    rws = data.reviews
    
    #sentences, token_lists, idx_in, mapeamento = preprocess_alt(rws, gabarito_preprocess, samp_size=int(args.samp_size)) #preprocesamento
    sentences, token_lists, idx_in, mapeamento = preprocessamento(rws, gabarito_preprocess)
    
    # Define the topic model object
    tm = Topic_Model(k = int(args.ntopic), method = str(args.method))
    
    # Fit the topic model by chosen method
    tm.fit(sentences, token_lists)
    
    # Evaluate using metrics
    #with open("../docs/saved_models/{}.file".format(tm.id), "wb") as f:
    #    pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)
        
    # Calculando as métricas
    coherence = get_coherence(tm, token_lists, 'c_v')
    silhouette_score = get_silhouette(tm)
    s1 = 'Coherence: ' + str(coherence)
    s2 = 'Silhouette Score: ' + str(silhouette_score)
    print(s1)
    print(s2)
    
    # visualize and save img
    visualize(tm)
        
    # Gerando as imagens dos grupos
    
    for i in range(tm.k):
        get_wordcloud(tm, token_lists, i)
        
    
    # Salvando as métricas
    dr = '../resultados/images/{}/{}'.format(tm.method, tm.id)
    if not os.path.exists(dr):
        os.makedirs(dr)
        
    arq = open('../resultados/images/{}/{}/'.format(tm.method, tm.id) + "metricas.txt","w")
    arq.write(s1 + '\n')
    arq.write(s2 + '\n')
    arq.close()
    
    '''
    if(args.method != "LDA"):
        #Obtendo as 20 palavras de maior frequencia de cada cluster
        k_words = get_topic_words(token_lists, tm.cluster_model.labels_)
        arq = open('../resultados/images/{}/{}/'.format(tm.method, tm.id) + "clusteres.txt", "w")
        arquivo = open('../resultados/images/{}/{}/'.format(tm.method, tm.id) + "k-topics-cluster.txt", "w")
        file_coments = open('../resultados/images/{}/{}/'.format(tm.method, tm.id) + "comentarios-topic.txt", "w")
        for i in range(tm.k):
            # Obtendo a nuvem de palavras do cluster
            #get_wordcloud(tm, token_lists, i, gabarito_preprocess)
            #get_wordcloud(tm, token_lists, i)
            
            # Obtendo os rótulos de cada documento
            lbs = tm.cluster_model.labels_
            # Documentos rotulados com o indice i
            coments = np.array(sentences)[lbs == i]
            # Quantidade de documentos selecionados
            tam = 5
            # Obtendo um número n de documentos
            comentarios = np.random.choice(coments,tam,replace=False)
            # Escrevendo num arquivo
            file_coments.write("Cluster {}:\n".format(i)) 
            # Salvando os comentários em um arquivo
            for ind in range(tam):
                file_coments.write(comentarios[ind] + '\n')
            file_coments.write('\n')
            
            top_topics = [] #Lista de palavras com maior frequência
            
            print(k_words[i])
            print(mapeamento)
            
            for k,v in k_words[i]:
                menor = min(mapeamento[k],key=lambda x: len(x))
                top_topics.append((menor,v))

            # Obtendo as palavras do cluster
            conjunto = sorted(get_words_topic(tm, token_lists, i))
            resp = set()
            # Obtendo a forma original das palavras
            for c in conjunto:
                resp = resp.union(mapeamento[c])
            resp = sorted(resp)
            # Salvando os resultados num arquivo
            saida = "Cluster {}: \n{}\n".format(i, str(resp))
            arq.write(saida + "\n")

            # Salvando as 20 palavras de maior frequencia no cluster
            palavras = [v[0] for v in top_topics]
            output = "Cluster {}: \n{}\n".format(i, str(palavras))
            arquivo.write(output + "\n")
        file_coments.close()
        arquivo.close()
        arq.close()
    '''