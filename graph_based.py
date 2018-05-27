import networkx as nx
import pickle
import glob
import numpy as np
from bulid_models import Logist_Reg_K_Fold
from bulid_models import NN_K_Fold
def create_vector(doc,word_list):
    vec=[]
    for i in range(0,len(word_list)):
        if(doc.__contains__(word_list[i])):
            vec.append(1)
        else:
            vec.append(0)
    return vec

def create_graph(file_input_dir, train_validation_set):
    data = []
    class_list = []
    G = nx.Graph()
    for filename in glob.glob(file_input_dir + '/*.txt'):
        if (filename in train_validation_set):
            file = open(filename, encoding="utf8", errors='ignore')
            filelines = file.readlines()
            str1 = ""
            for each in filelines:
                str1 = str1 + each
            data.append(str1)
            split_string=str1.split(" ")
            G.add_edge(split_string[0],split_string[1])
            for i in range(1,len(split_string)-1):
                if(split_string[i]!=split_string[i-1]):
                    G.add_edge(split_string[i-1], split_string[i])
                if(split_string[i]!=split_string[i+1]):
                    G.add_edge(split_string[i],split_string[i+1])
            if (filename.__contains__('ham')):
                temp = []
                temp.append(1)
                class_list.append(temp)
            else:
                temp = []
                temp.append(0)
                class_list.append(temp)


if __name__ == "__main__":
        class_list = []
        a = []
        train_validation_set = pickle.load(open('train_validation_sets.txt', 'rb'))
        create_graph('raw_input', train_validation_set)