import nltk
nltk.download('stopwords')

from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from nltk.corpus import stopwords
from tqdm import tqdm

#Função que retorna a sentenca
def get_Sentence(s):
    resp = ""
    for i in range(len(s)):
        if(s[i] == '>'):
            resp = s[i+1:]
            break
    resp = resp.strip()
    return resp

#Diretorios dos datasets
caminhos = ['cameras.xml','cells.xml','dvds.xml','laptops.xml','routers.xml']
#Diretorios dos arquivos das sentencas
save_sentencas = ['sentencas-camera.txt','sentencas-cells.txt','sentencas-dvds.txt','sentencas-laptops.txt','sentencas-routers.txt']
#Diretorios dos arquivos de palavras-sentencas
save_palavras = ['palavras-camera.txt','palavras-cells.txt','palavras-dvds.txt','palavras-laptops.txt','palavras-routers.txt']
#Diretorios dos arquivos de frequencias de palavras
save_frequencias = ['frequencias-camera.txt','frequencias-cells.txt','frequencias-dvds.txt','frequencias-laptops.txt','frequencias-routers.txt']

#Para cada dominio
for i in range(len(caminhos)):
#for i in range(1):
    sentencas = [] #Variavel que guarda as sentencas do documento
    documentos = [] #Variavel que guarda os documentos do dominio
    palavras = {} #Variavel que mapeia as sentencas para as palavras
    sentenca = ""
    palavra = ""
    #Acessando os datasets
    arq = open('../../datasets/' + caminhos[i],'r')
    #Percorrendo as linhas do arquivo
    for linha in tqdm(arq.readlines(), desc='Leitura Dados'):
        if('<sentence' in linha):
            s = linha.replace('</sentence>','')
            s = s.split('idSentence=')[1]
            s = get_Sentence(s) #Obtendo a sentenca
            sentenca = s
            sentencas.append(sentenca)
        if('<opinion>' in linha):
            p = linha.replace('/opinion','opinion')
            p = p.replace('<opinion>','')
            p = p.split('"')
            palavra = p[1].lower() #Obtendo o opinion (atributo)
            tipo = p[-2]
            if(tipo != 'anaphora' and (palavra not in set(stopwords.words('english')))):
                if(palavra not in palavras.keys()):
                    palavras[palavra] = set()
                palavras[palavra].add(sentenca)
        if('</review>' in linha):
            sentenca = ""
            palavra = ""
            documentos.append(sentencas)
            sentencas = []
    arq.close()

    #Salvando as sentencas num arquivo
    arq = open('../../datasets_processed/sentencas/' + save_sentencas[i],'w')
    for doc in tqdm(documentos, desc='Setencas'):
        for sent in doc: 
            arq.write(sent + '\n')
        arq.write('\n')
    arq.close()

    #Salvando as palavras com suas sentencas nos arquivos
    arq = open('../../datasets_processed/' + save_palavras[i],'w')
    for key in tqdm(sorted(palavras.keys()), desc='Palavras e Setencas'):
        arq.write(key + ' ->> ') #Caracter separador de palavra e sentença
        arq.write(str(palavras[key])+'\n')
    arq.close()

    #Salvando as sentenças numa variavel
    arq = open('../../datasets_processed/sentencas/' + save_sentencas[i],'r')
    texto = arq.read().lower() #Obtem todos os documentos
    arq.close()

    #Salvando as frequencias nos arquivos
    arq = open('../../datasets_processed/frequencias/' + save_frequencias[i],'w')
    for p in tqdm(sorted(palavras.keys()), desc='Frequencias'):
        frequencia = 0
        '''if(' ' in p):
            frequencia = texto.count(p)
        else:
            frequencia = TextBlob(texto).words.count(p)'''
        frequencia = texto.count(p)
        arq.write(p + ': ' + str(frequencia) + '\n')
    arq.close()

