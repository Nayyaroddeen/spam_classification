from bulid_models import preprocess_docs
import pickle
from sklearn.metrics import accuracy_score
from bulid_models import count_vector_model
def transfrom_word_cound_model(data):
    count_vec=pickle.load(open('best_models/count_vec_model.sav', 'rb'))
    return count_vec.transform(data)


def test_logist_model(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


if __name__ == "__main__":
    class_list=[]
    a=[]
    test_set_docs=pickle.load(open('test_set.txt', 'rb'))
    count_vec,class_list=preprocess_docs('raw_input',test_set_docs)
    count_vec=transfrom_word_cound_model(count_vec)
    test_logist_model(count_vec.toarray().astype(int), class_list)