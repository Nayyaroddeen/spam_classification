import networkx as nx
import pickle
import glob
import numpy as np
from bulid_models import Logist_Reg_K_Fold
from bulid_models import NN_K_Fold
from bulid_models import Random_Forest_K_Fold
from bulid_models import CNN_K_Fold
''' This function creates feature vector using the word list'''
def create_vector(doc,word_list):
    vec=[]
    for i in range(0,len(word_list)):
        if(doc.__contains__(word_list[i])):
            vec.append(1)
        else:
            vec.append(0)
    return vec

''' This function creates graph using the networkxlibrary by using only
one word window and saves the key core features and coloring features based on the greedy algortihm
'''

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
            #graph creation using one word window
            G.add_edge(split_string[0],split_string[1])
            for i in range(1,len(split_string)-1):
                if(split_string[i]!=split_string[i-1]):
                    G.add_edge(split_string[i-1], split_string[i])
                if(split_string[i]!=split_string[i+1]):
                    G.add_edge(split_string[i],split_string[i+1])
            #creating the class label
            if (filename.__contains__('ham')):
                temp = []
                temp.append(1)
                class_list.append(temp)
            else:
                temp = []
                temp.append(0)
                class_list.append(temp)
    #extracting the core features using the kcore algorithm
    k_core_feature_list=list(nx.k_core(G).nodes())
    #colorng the created graph
    coloring_dict = nx.coloring.greedy_color(G)
    #saving the core features
    filename = 'k_core_features.sav'
    pickle.dump(k_core_feature_list, open('best_models/' + filename, 'wb'))

    print(len(k_core_feature_list))
    val = []
    keys = []
    #saving the vertices more than the color 20
    for key, item in coloring_dict.items():
        if (item > 20):
            keys.append(key)
    filename = 'coloring_features.sav'
    pickle.dump(keys, open('best_models/' + filename, 'wb'))

''' building the models using the core features once the input is scanned then
 then we pass each document to create the core based features to get the new features
 then we use this features to build the models'''
def build_models_using_core(file_input_dir,train_validation_set):
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
            if (filename.__contains__('ham')):
                temp = []
                temp.append(1)
                class_list.append(temp)
            else:
                temp = []
                temp.append(0)
                class_list.append(temp)
    vec_array=[]
    word_list = pickle.load(open('best_models/k_core_features.sav', 'rb'))

    for i in range(0, len(data)):
        temp_list = create_vector(data[i], word_list)
        vec_array.append(temp_list)
    vec_array=np.array(vec_array)
    class_list=np.array(class_list)
    Logist_Reg_K_Fold(vec_array,class_list)
    Random_Forest_K_Fold(vec_array,class_list)
    NN_K_Fold(vec_array, class_list)
    CNN_K_Fold(vec_array, class_list)


''' building the models using the coloring features once the input is scanned then
 then we pass each document to create the coloring based features to get the new features
 then we use this features to build the models'''

def build_models_using_coloring(file_input_dir,train_validation_set):
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
            if (filename.__contains__('ham')):
                temp = []
                temp.append(1)
                class_list.append(temp)
            else:
                temp = []
                temp.append(0)
                class_list.append(temp)
    vec_array=[]
    word_list = pickle.load(open('best_models/coloring_features.sav', 'rb'))

    for i in range(0, len(data)):
        temp_list = create_vector(data[i], word_list)
        vec_array.append(temp_list)
    vec_array=np.array(vec_array)
    class_list=np.array(class_list)
    Logist_Reg_K_Fold(vec_array,class_list)
    Random_Forest_K_Fold(vec_array,class_list)
    NN_K_Fold(vec_array, class_list)
    CNN_K_Fold(vec_array, class_list)
if __name__ == "__main__":
        class_list = []
        a = []
        #loading the training file names
        train_validation_set = pickle.load(open('train_validation_sets.txt', 'rb'))
        '''this function is used to create the graph based features'''
        create_graph('raw_input', train_validation_set)
        ''' this functions builds all the models using core features'''
        build_models_using_core('raw_input', train_validation_set)
        ''' this functions builds all the models using coloring features'''
        build_models_using_coloring('raw_input', train_validation_set)