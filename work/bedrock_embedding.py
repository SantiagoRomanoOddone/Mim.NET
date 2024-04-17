from langchain_community.embeddings import BedrockEmbeddings
from numpy import dot
from numpy.linalg import norm
from joblib import load
import numpy as np

#create an Amazon Titan Embeddings client
belc = BedrockEmbeddings()



class EmbedItem:
    def __init__(self, text):
        self.text = text
        self.embedding = belc.embed_query(text)

class ComparisonResult:
    def __init__(self, text, similarity):
        self.text = text
        self.similarity = similarity

def calculate_similarity(a, b): 
    return dot(a, b) / (norm(a) * norm(b))


def finding_best_match(input_file, input_file_2 ,case):
    items = []
    result = []
    similarities = []
    for text in input_file:
        items.append(EmbedItem(text))
    
    cosine_comparisons = []
        
    for e2 in items:
        similarity_score = calculate_similarity(EmbedItem(case).embedding, e2.embedding)
        cosine_comparisons.append(ComparisonResult(e2.text, similarity_score)) #save the comparisons to a list
        
    cosine_comparisons.sort(key=lambda x: x.similarity, reverse=True) # list the closest matches first
    
    closest_matches = ""
    for c in cosine_comparisons[:3]:  
        #similarities.append(f"%.6f" % c.similarity + "\t" + c.text)
        closest_matches = c.text
        #similarities.append(c.similarity)
        result.append(input_file_2[input_file.index(closest_matches)])
        
    #result = f"{case} , el mismo tiene un comportamiento similar a los siguientes usuarios: \n{closest_matches}"  # Concatenate the case string with the closest matches
    return result


def get_success_events(results): 
    paths = load('/home/ubuntu/environment/workshop/mim.net/paths_of_session')
    
    paths_to_analyze = []
    for i in range(len(paths)):
        if paths[i]['user_id'] in results:
            paths_to_analyze.append(paths[i])
            
    if len(paths_to_analyze) > 3: 
        paths_to_analyze =  np.random.choice(paths_to_analyze, size = 3, replace = False)
    
    for i in paths_to_analyze:
        del i['user_id']
        
    return paths_to_analyze
