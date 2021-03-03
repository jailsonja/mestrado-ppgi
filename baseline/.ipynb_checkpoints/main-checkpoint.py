from model.cafe import Cafe
from utils.utils import set_candidate_terms, reviews_frequencies, read_reviews_txt


print("Iniciando....")

def main():
    # ler conjunto de termos candidatos
    candidates = set_candidate_terms('Cell-phones')
    model = Cafe(candidates)
    
    # frequencia de reviews
    frequencie_terms, matrix_terms = reviews_frequencies('Cell-phones')
    #print(len(frequencie_terms))
    #print()
    print(matrix_terms['able'])
    
    

main()