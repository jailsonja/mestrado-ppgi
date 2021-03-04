from model.cafe import Cafe
from utils.utils import set_candidate_terms, reviews_frequencies, read_reviews_txt, get_numbers_documents, read_setences_terms
from preprocess.preprocess import get_matrixG, generate_matriz, get_matrixT

print("Iniciando....")

def main():
    # ler conjunto de termos candidatos
    print("Leitura Candidatos Termos")
    candidates = set_candidate_terms('Cell-phones')
    setences_terms = read_setences_terms('Cell-phones', candidates))
    
    print("Instancia modelo")
    #model = Cafe(candidates)
    
    # frequencia de reviews
    #frequencie_terms, matrix_terms = reviews_frequencies('Cell-phones')
    #n = get_numbers_documents('Cell-phones')
    #print(frequencie_terms)
    #print(model.most_frequent_terms(frequencie_terms))
    #generate_matriz(candidates)
    
    #mtxT = get_matrixT(candidates, frequencie_terms, matrix_terms, n)
    #print(mtxT["quality"]["quality"])
    #print(get_matrixG(candidates).keys())
    

main()