from sklearn.feature_extraction.text import CountVectorizer
import glob
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
def count_vector_model(list_of_docs):
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(list_of_docs)
    filename = 'count_vec_model.sav'
    pickle.dump(count_vect, open('best_models/'+filename, 'wb'))
    return train_counts


def tf_idf_model(list_of_docs):
    tfidf_transformer = TfidfVectorizer()
    train_tfidf = tfidf_transformer.fit_transform(list_of_docs)
    filename = 'tfidf_vec_model.sav'
    pickle.dump(tfidf_transformer, open('best_models/'+filename, 'wb'))
    return train_tfidf



def preprocess_docs(file_input_dir,train_validation_set):
    data=[]
    class_list=[]
    for filename in glob.glob(file_input_dir+'/*.txt'):
        if(filename in train_validation_set):
            file=open(filename ,encoding="utf8", errors='ignore')
            filelines=file.readlines()
            str1=""
            for each in filelines:
                str1=str1+each
            data.append(str1)
            if(filename.__contains__('ham')):
                temp=[]
                temp.append(1)
                class_list.append(temp)
            else:
                temp = []
                temp.append(0)
                class_list.append(temp)
    return data,class_list

def Logist_Reg_K_Fold(data,class_list):
    seed = 7
    kfold = StratifiedKFold(n_splits=5
                            , shuffle=True, random_state=seed)
    base_acc=0
    for train, validation in kfold.split(data, class_list):
        train_data=data[train]
        train_class=class_list[train]

        validation_data=data[validation]
        validation_class=class_list[validation]

        logisticRegr = LogisticRegression()
        logisticRegr.fit(train_data.toarray().astype(int), train_class)
        x_test = logisticRegr.predict(validation_data.toarray().astype(int))
        validation_acc=accuracy_score(validation_class, x_test)
        if(base_acc<validation_acc):
            filename = 'logistic_best_model.sav'
            pickle.dump(logisticRegr, open('best_models/' + filename, 'wb'))
        print(accuracy_score(validation_class, x_test))

if __name__ == "__main__":
    class_list=[]
    a=[]
    train_validation_set=pickle.load(open('train_validation_sets.txt', 'rb'))
    count_vec,class_list=preprocess_docs('raw_input',train_validation_set)
    count_vec=count_vector_model(count_vec)
    Logist_Reg_K_Fold(count_vec, np.array(class_list))



