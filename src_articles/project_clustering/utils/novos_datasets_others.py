import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#Bibliotecas utilizadas no notebook
import pandas as pd
import numpy as np
from textblob import TextBlob
import ast
from nltk.corpus import stopwords
import spacy #Importando a biblioteca do spacy
import gensim
import ast
import os
import math

# Função que retorna se um termo está escrito correto
# str.isalpha: verifica se na sentenca ocorre somente caracteres alfabeticos
def isCorrect(sentenca):
    for c in sentenca:
        if(c != ' '):
            if(not c.isalpha()): 
                return False
    return True

# Função que retorna a frequência de uma palavra única
def checagem_frequencia(word, sentenca):
    tokens = TextBlob(sentenca).words #Quebra a sentenca em palavras
    count = 0 #Frequencia do termo
    for palavra in tokens:
        if(word.lower() in palavra.lower()): #Verifica se a palavra corresponde
            count += 1
    return count

# Função que retorna a frequência de um termo em uma sentença
def get_frequencia(word,sentenca):
    count = 0 #Frequencia da palavra
    if(' ' in word): #Se for uma palavra composta
        word = word.lower()
        sentenca = sentenca.lower()
        count = sentenca.count(word)
    else: #Se for uma palavra única
        count = checagem_frequencia(word,sentenca)
    return count  

def isNoun(word):
    if(' ' in word):
        return True
    tags = TextBlob(word).pos_tags
    if('NN' in tags[0][1]):
        return True
    return False

def contexto(p1,p2,mapeamento):
    fx = 0 #Variavel que guarda o numero de documentos que p1 aparece
    fy = 0 #Variavel que guarda o numero de documentos que p2 aparece
    fxy = 0 #VAriavel que guarda o numero de documentos que p1 e p2 aparecem em uma sentenca
    n = len(mapeamento[p1]) #Numero de documentos
    #Percorrendo os documentos
    for i in list(range(n)):
        if(len(mapeamento[p1][i].intersection(mapeamento[p2][i])) > 0):
            fxy += 1
        if(len(mapeamento[p1][i]) > 0):
            fx += 1
        if(len(mapeamento[p2][i]) > 0):
            fy += 1
    #Obtendo o NPMI
    npmi = 0
    fator1 = (n*fxy)/(fx*fy)
    fator2 = (fxy/n)
    try:
        valor = math.log10(fator1)/-math.log10(fator2)
        npmi = (valor+1)/2 #Normalizando para o intervalo [0,1]
    except:
        npmi = 0.0
    return npmi

#Diretorio dos datasets
save_mapeamentos = ['../../datasets_processed/mapeamentos/mapeamentos-camera-others.txt','../../datasets_processed/mapeamentos/mapeamentos-cell-others.txt']

save_palavras = ['../../datasets_processed/palavras-camera-others.txt','../../datasets_processed/palavras-cell-others.txt']

save_frequencias = ['../../datasets_processed/frequencias/frequencias-camera-others.txt','../../datasets_processed/frequencias/frequencias-cell-others.txt']

save_similaridades = ['../../datasets_processed/similaridades/similaridades-camera-others.txt','../../datasets_processed/similaridades/similaridades-cell-others.txt']

save_contextos = ['../../datasets_processed/contextos/contexto-camera-others.txt','../../datasets_processed/contextos/contexto-cell-others.txt']

save_sentencas = ['../../datasets_processed/sentencas/sentencas-camera-others.txt','../../datasets_processed/sentencas/sentencas-cell-others.txt']

save_contextualizacao = ['../../datasets_processed/contextualizacao/contextualizacao-camera-others.txt','../../datasets_processed/contextualizacao/contextualizacao-cell-others.txt']

save_csv = ['../../dataset_outros/dataset-camera-others.csv','../../dataset_outros/dataset-cell-others.csv']

save_datasets = ['../../datasets/dados-camera-others','../../datasets/dados-cell-others']

# inseri os caminhos dos arquivos dos direórios em uma lista
caminhos = ['../../dataset_outros/{}'.format(file) for file in os.listdir('../../dataset_outros/') if '.tsv' in file]
dir_gabaritos = ['../../datasets/gabaritos/{}'.format(file) for file in os.listdir('../../datasets/gabaritos/') if '.txt' in file]

# ordena as listas em ordem alfabética
dir_gabaritos = sorted(dir_gabaritos)
caminhos = sorted(caminhos)

dominio = {} # Dicionário que indica o indice para os arquivos de dados de cada produto
lista_produtos = [file.replace('.txt', '') for file in os.listdir('../../datasets/gabaritos/') if '.txt' in file] # lista produtos

for i, value in enumerate(dir_gabaritos):
    temp = value.split('/')[-1].replace('.txt', '')
    dominio[temp] = i
    
lista_produtos = sorted(lista_produtos)

dados = {}
stop_words = stopwords.words('english')
frequencias = {}
documentos = set()
gabarito = set()

for produto in lista_produtos[:2]:
    arq = open(dir_gabaritos[dominio[produto]],'r') # gabarito de um determinado produto
    for linha in arq.readlines():
        valores = linha.split(': ')
        classe = valores[0]
        if(classe != "Others"): # classes diferentes de outros, verifica a gramatica lexical e adiciona no conjunto
            atributes = ast.literal_eval(valores[1])
            gabarito = gabarito.union(atributes)
    arq.close()
    
    df = pd.read_table(caminhos[dominio[produto]],names=['aspecto','sentenca','sla'])
    
    df = df.drop_duplicates() #removendo linhas duplicadas do dataset
    
    df = df.drop("sla",axis=1) #removendo a última coluna do dataset
    
    df['is_correct'] = df['aspecto'].apply(lambda x: isCorrect(x))
    
    palavras_corretas = df[df['is_correct'] == True]
    
    palavras_corretas['is_stopword'] = palavras_corretas['aspecto'].apply(lambda x: x in stop_words)
    palavras_corretas['is_other'] = palavras_corretas['aspecto'].apply(lambda x: x not in gabarito)
    palavras_corretas = palavras_corretas.query('is_stopword == False & is_other == True')
    
    palavras_corretas['isNN'] = palavras_corretas['aspecto'].apply(lambda x: isNoun(x))
    
    palavras_corretas = palavras_corretas.query("isNN == True")
    
    database = palavras_corretas[['aspecto','sentenca']]#.sample(n=6000,random_state=42)
    
    database_sample = database.sample(n=5000,random_state=42)
    
    dados_teste_camera_others = database.sample(n=5000,random_state=42)
    dados_teste_camera_others = dados_teste_camera_others.rename(columns={"sentenca": 'reviews'})
    dados_teste_camera_others.to_csv(save_datasets[dominio[produto]]+str(len(dados_teste_camera_others))+".csv",index=False)
    
    aspectos_selecionados = database_sample['aspecto'].value_counts().index[:200]
    aspectos_escolhidos = list(aspectos_selecionados)
    dataset_refinado = database_sample[database_sample['aspecto'].isin(aspectos_escolhidos)]
    
    dataset_refinado.to_csv(save_csv[dominio[produto]],index=False)
    
    print("---------PRE PROCESSAMENTO-----------")
    
    dataframe = pd.read_csv(save_csv[dominio[produto]])
    datasample = dataframe
    
    #Lendo o dataset - obtendo os aspectos corretos e suas sentenças
    tam = len(datasample)
    for ind in range(tam):
        word = datasample.aspecto[ind].strip()
        sentenca = datasample.sentenca[ind].strip()
        count_frequencia = get_frequencia(word,sentenca)
        if(count_frequencia > 0):
            if(word not in dados.keys()):
                dados[word] = set()
                frequencias[word] = 0
            #Caso tudo esteja correto eu pego a palavra e o documento que ela aparece
            dados[word].add(sentenca)            
            documentos = documentos.union(set([sentenca]))
            frequencias[word] += count_frequencia
            
    # Calculando as frequencias dos aspectos
    valor_frequencias = {}
    for key in sorted(frequencias.keys()):
        freq = frequencias[key]
        if(freq not in valor_frequencias.keys()):
            valor_frequencias[freq] = []
        valor_frequencias[freq].append(key)
        
    # Selecionando as 100 palavras de maior frequência
    cont = 0
    palavras_utilizadas = []
    documentos_utilizados = set()
    for valor in sorted(valor_frequencias.keys(),reverse=True):
        if(cont < 150):
            for palavra in valor_frequencias[valor]:
                palavras_utilizadas.append(palavra)
                documentos_utilizados = documentos_utilizados.union(dados[palavra])
                cont += 1
                
    documentos_utilizados = sorted(documentos_utilizados) #Ordenando os documentos
    palavras_utilizadas.sort() #Ordenando as termos
    
    documents = documentos_utilizados#[:2000] #Selecionando os primeiros 2000 documentos do dataset
    
    mapeamentos = {} #Variável que mapeia os domentos para os termos
    frequencias = {} #Variável que mapeia as frequências para os termos
    for doc in documents:
        for p in palavras_utilizadas:
            count = get_frequencia(p,doc)
            if(count > 0):
                if(p not in mapeamentos.keys()):
                    mapeamentos[p] = set()
                if(p not in frequencias.keys()):
                    frequencias[p] = 0
                mapeamentos[p].add(doc) #Mapeando as sentenças para os termos
                frequencias[p] += count #Obtendo as frequências dos termos nos documentos
                
    print("--------- SALVANDO OS DADOS EM ARQUIVOS --------------")
    
    # Salvando as palavras selecionadas em um arquivo
    arq = open(save_palavras[dominio[produto]],'w')
    for p in palavras_utilizadas:
        arq.write(p + ' ->> ' + str(mapeamentos[p]) + '\n')
    arq.close()
    
    # Salvando as frequências em um arquivo
    arq = open(save_frequencias[dominio[produto]],'w')
    for p in palavras_utilizadas:
        arq.write(p + ': ' + str(frequencias[p]) + '\n')
    arq.close()
    
    # Salvando os documentos em um arquivo
    arq = open(save_sentencas[dominio[produto]],'w')
    for doc in documents:
        arq.write(doc + '\n\n')
    arq.close()
    
    # Salvando os mapeamentos em um arquivo
    arq = open(save_mapeamentos[dominio[produto]],'w')
    for p in palavras_utilizadas:
        arq.write(p + ' ->> ' + str(mapeamentos[p]) + '\n')
    arq.close()
    
    print("------------ PRE PROCESSAMENTO CONTEXTUALIZAÇÂO ----------------")
    
    n = len(documents) #Número de documentos utilizados
    n_palavras = len(palavras_utilizadas) #Número de palavras utilizadas no processo
    contextualizacao = {} #Variável que salva os documentos que cada termo aparece
    
    # Inicializando a variável - uso esta variável para calcular o contexto entre as palavras
    for p in palavras_utilizadas:
        contextualizacao[p] = []
        
    # Fazendo o mapeamento dos documentos para os termos
    for p in palavras_utilizadas:
        for doc in documents:
            count = get_frequencia(p,doc)
            if(count > 0):
                contextualizacao[p].append([doc])
            else:
                contextualizacao[p].append([])
                
    # Salvando a variável de contextualização
    arq = open(save_contextualizacao[dominio[produto]],'w')
    for p in sorted(contextualizacao.keys()):
        arq.write(p + ' ->> ' + str(contextualizacao[p]) + '\n')
    arq.close()
    
    print("----------- OBTENDO OS DADOS DOS ARQUIVOS ------------")
    
    # Variáveis utilizadas
    palavras_utilizadas = []
    documentos_utilizados = []
    contextualizacao = {}
    
    # Obtendo a variável responsável pela contextualização das palavras
    arq = open(save_contextualizacao[dominio[produto]],'r')
    for linha in arq.readlines():
        valores = linha.split(' ->> ')
        classe = valores[0]
        conjunto = valores[1]
        palavras_utilizadas.append(classe)
        contextualizacao[classe] = ast.literal_eval(conjunto)
    arq.close()
    
    # Obtendo os documentos
    arq = open(save_sentencas[dominio[produto]],'r')
    for linha in arq.readlines():
        if(linha != '\n'):
            documentos_utilizados.append(linha.strip())
    arq.close()
    
    n_palavras = len(palavras_utilizadas) #Número de palavras(aspectos)
    n = len(documentos_utilizados) #Número de documentos
    
    print("---------- CALCULANDO SIMILARIDADE ---------------")
    
    matrizG = [[1.0]*n_palavras for i in range(n_palavras)] #Matriz de similaridade
    # Locais onde salvar as similaridades
    caminhoG = ['../../datasets_processed/similaridades/similaridades-camera-others.txt','../../datasets_processed/similaridades/similaridades-cell-others.txt']
    
    nlp = spacy.load("en_vectors_web_lg") #Carregando o modelo do spacy
    print("Processando Spacy...")
    #Salvando as similaridades num arquivo
    arq = open(caminhoG[dominio[produto]],'w')
    for ind in range(len(palavras_utilizadas)):
        token1 = nlp(palavras_utilizadas[ind]) #Palavra 1
        for j in range(ind+1,len(palavras_utilizadas)):
            token2 = nlp(palavras_utilizadas[j]) #Palavra 2
            try:
                similaridade = token1.similarity(token2) #Similaridade entre a palavra 1 e palavra 2
                if(similaridade < 0):
                    similaridade = 0.0
                if(similaridade > 1.0):
                    similaridade = 1.0
            except:
                similaridade = 0.0
            #Salvando a similaridade no arquivo
            arq.write(palavras_utilizadas[ind]+'->>'+palavras_utilizadas[j]+'->>'+str(similaridade)+'\n')
            matrizG[ind][j] = similaridade
            matrizG[j][ind] = similaridade
    arq.close()
    print("Concluido!\n")
    
    caminho_word2vec = ['../../datasets_processed/similaridades/similaridades-camera-others-word2vec.txt','../similaridades/similaridades-cell-others-word2vec.txt']
    matriz_word2vec = [[0.0]*n_palavras for i in range(n_palavras)]
    caminho = '../../model/GoogleNews-vectors-negative300.bin'
    
    #Carregando o modelo do Word2Vec
    model = gensim.models.KeyedVectors.load_word2vec_format(caminho,binary=True)
    
    #Salvando as similaridades do Word2Vec num arquivo
    arq = open(caminho_word2vec[dominio[produto]], 'w')
    for ind in range(len(palavras_utilizadas)):
        for j in range(ind+1, len(palavras_utilizadas)):
            try:
                #Obtendo a similaridade usando o modelo
                similaridade = model.similarity(palavras_utilizadas[ind], palavras_utilizadas[j])
                if(similaridade < 0):
                    similaridade = 0.0
                if(similaridade > 1.0):
                    similaridade = 1.0
            except:
                similaridade = 0.0
            matriz_word2vec[i][j] = similaridade
            matriz_word2vec[j][i] = similaridade
            
            #Escrevendo o resultado no arquivo
            arq.write(palavras_utilizadas[ind].replace('_', ' ')+'->>'+palavras_utilizadas[j].replace('_', ' ')+'->>'+str(similaridade)+'\n')
    arq.close()
    print("Concluido!\n")
    
    print("------------ CALCULANDO CONTEXTO ENTRE PALAVRAS --------------")
    # Obtendo a contextualização das palavras
    mapeamento = {}
    for p in sorted(contextualizacao.keys()):
        vetor = contextualizacao[p]
        vetor = [set(v) for v in vetor]
        mapeamento[p] = vetor
        
    matrizT = [[1.0]*n_palavras for i in range(n_palavras)] #Matriz de contextualização
    
    tam = n_palavras #Quantidade de palavras no gabarito
    total = (1+tam)*tam/2 #Total de iteracoes que serão feitas
    cont = 0 #Variavel contadora
    percent = 0 #Valor da porcentagem do processamento
    aux = 0 #Variavel auxiliar
    #Salvando o valor do NPMI entre as palavras
    arq = open(save_contextos[dominio],'w')
    print('Calculando o contexto entre as palavras...')
    for i in tqdm(range(n_palavras)):
        for j in range(i+1,n_palavras):
            cont += 1
            #valor = NPMI(words[i],words[j],documentos)
            valor = contexto(palavras_utilizadas[i],palavras_utilizadas[j],mapeamento) #Valor do NPMI
            matrizT[i][j] = valor
            matrizT[j][i] = valor
            resp = str(palavras_utilizadas[i]) + '->>' + str(palavras_utilizadas[j]) + '->>' + str(valor) + '\n'
            arq.write(resp) #Salvando as palavras num arquivo
            #Printando o valor da porcentagem na tela
            aux = int(cont*100/total)
            if(aux > percent):
                #print(str(aux) + "% concluido")
                percent = aux
    arq.close()
    print("Concluido\n")
    
    print("---------- GERANDO E SALVANDO MATRIZES --------------")
    indices = {} #Variável que mapeia os termos para os seus indices nas matrizes
    ind = 0
    for i in range(n_palavras):
        word = palavras_utilizadas[i]
        indices[word] = ind
        ind += 1
        
    # Obtendo as similaridades entre os termos
    arq = open(caminhoG[dominio[produto]],'r')
    for linha in arq.readlines():
        valores = linha.split('->>')
        p1 = valores[0]
        p2 = valores[1]
        valor = float(valores[2])
        x = indices[p1]
        y = indices[p2]
        #Colocando os valores das similaridades nas matrizes
        matrizG[x][y] = valor
        matrizG[y][x] = valor
    arq.close()
    
    # Obtendo o NPMI entre as palavras
    arq = open(save_contextos[dominio[produto]],'r')
    for linha in arq.readlines():
        valores = linha.split('->>')
        p1 = valores[0]
        p2 = valores[1]
        valor = float(valores[2])
        x = indices[p1]
        y = indices[p2]
        #Salvando o NPMI na matriz de contextualização
        matrizT[x][y] = valor
        matrizT[y][x] = valor
    arq.close()
    
    # Diretórios dos locais para salvar as matrizes de similaridade e contexto
    save_matrizG = ['../../datasets_processed/matrizes/similaridades-camera-others.txt','../../datasets_processed/matrizes/similaridades-cell-others.txt']
    save_matrizT = ['../../datasets_processed/matrizes/contextos-camera-others.txt','../../datasets_processed/matrizes/contextos-cell-others.txt']
    
    # Salvando a matriz de similaridade
    arq = open(save_matrizG[dominio[produto]],'w')
    for vetor in matrizG:
        arq.write(str(vetor) + '\n')
    arq.close()
    
    # Salvando a matriz de contexto
    arq = open(save_matrizT[dominio[produto]],'w')
    for vetor in matrizT:
        arq.write(str(vetor) + '\n')
    arq.close()