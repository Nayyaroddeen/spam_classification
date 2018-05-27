from bulid_models import preprocess_docs
import pickle
from sklearn.metrics import accuracy_score
from bulid_models import count_vector_model
from keras.models import model_from_json
import numpy as np
def transfrom_word_cound_model(data):
    count_vec=pickle.load(open('best_models/count_vec_model.sav', 'rb'))
    return count_vec.transform(data)


def test_logist_model(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))

def test_random_forest_model(data,class_list):

    logist_model=pickle.load(open('best_models/random_forest_best_model.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))

def test_nn_model(data,class_list):

    json_file = open('best_models/nn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/nn_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = loaded_model.predict(data)
    predict_out = []
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)

def test_cnn_model(data,class_list):
    data = np.expand_dims(data, axis=2)
    json_file = open('best_models/cnn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/cnn_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = loaded_model.predict(data)
    predict_out = []
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)

if __name__ == "__main__":
    class_list=[]
    a=[]
    test_set_docs=pickle.load(open('test_set.txt', 'rb'))
    count_vec,class_list=preprocess_docs('raw_input',test_set_docs)
    count_vec=transfrom_word_cound_model(count_vec)
    #stest_logist_model(count_vec.toarray().astype(int), class_list)
    #test_random_forest_model(count_vec.toarray().astype(int), class_list)
    #test_nn_model(count_vec.toarray().astype(int),class_list)
    test_cnn_model(count_vec.toarray().astype(int), class_list)
