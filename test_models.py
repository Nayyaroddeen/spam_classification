from bulid_models import preprocess_docs
import pickle
from sklearn.metrics import accuracy_score
from bulid_models import count_vector_model
from keras.models import model_from_json
import numpy as np
from graph_based import create_vector
''' This function returns the word count vector for each of the docs'''
def transfrom_word_cound_model(data):
    count_vec=pickle.load(open('best_models/count_vec_model.sav', 'rb'))
    return count_vec.transform(data)

''' This function returns the tfidf for each of the docs'''

def transfrom_word_tfidf(data):
    tfidf=pickle.load(open('best_models/tfidf_vec_model.sav', 'rb'))
    return tfidf.transform(data)

''' This function returns the core feature vector of each of the docs'''

def transfrom_kcore(data):
    vec_array=[]
    word_list = pickle.load(open('best_models/k_core_features.sav', 'rb'))

    for i in range(0, len(data)):
        temp_list = create_vector(data[i], word_list)
        vec_array.append(temp_list)
    vec_array=np.array(vec_array)

    return vec_array

''' This function returns the coloring based feature vector of each of the docs'''
def transfrom_coloring(data):
    vec_array=[]
    word_list = pickle.load(open('best_models/coloring_features.sav', 'rb'))

    for i in range(0, len(data)):
        temp_list = create_vector(data[i], word_list)
        vec_array.append(temp_list)
    vec_array=np.array(vec_array)

    return vec_array
''' This functions loads the pre trained model and test and prints the accuracy for each 
for the testing data using word count feature vectors'''
def test_logist_model(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))

''' This functions loads the pre trained model and test and prints the accuracy for each 
for the testing data using tfidf'''
def test_logist_model_tfidf(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model_tfidf.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))

''' This functions loads the pre trained model and test and prints the accuracy for each 
for the testing data using kcore features'''
def test_logist_model_kcore(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model_kcore.sav', 'rb'))
    data=transfrom_kcore(data)
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


''' This functions loads the pre trained model and test and prints the accuracy for each 
for the testing data using coloring based features'''
def test_logist_model_coloring(data,class_list):

    logist_model=pickle.load(open('best_models/logistic_best_model_coloring.sav', 'rb'))
    data=transfrom_coloring(data)
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


''' This functions loads the pre trained model of random forest and test and prints the accuracy for each 
for the testing data using kcore features'''
def test_random_forest_kcore(data,class_list):

    logist_model=pickle.load(open('best_models/random_forest_best_model_kcore.sav', 'rb'))
    data=transfrom_kcore(data)
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


''' This functions loads the pre trained model of random forest and test and prints the accuracy for each 
for the testing data using coloring based features'''

def test_random_forest_coloring(data,class_list):

    logist_model=pickle.load(open('best_models/random_forest_best_model_coloring.sav', 'rb'))
    data=transfrom_coloring(data)
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


''' This functions loads the pre trained model of random forest and test and prints the accuracy for each 
for the testing data using word count features'''

def test_random_forest_model(data,class_list):

    logist_model=pickle.load(open('best_models/random_forest_best_model.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))


''' This functions loads the pre trained model of random forest and test and prints the accuracy for each 
for the testing data using tfidf'''

def test_random_forest_tfidf(data,class_list):

    logist_model=pickle.load(open('best_models/random_forest_best_model_tfidf.sav', 'rb'))
    y_out=logist_model.predict(data)
    print(accuracy_score(class_list,y_out))
    print(len(data))
''' This functions loads the pre trained model of nerual network and test and prints the accuracy for each 
for the testing data using word count vector'''

def test_nn_model(data,class_list):
    #loading the pretrained model
    json_file = open('best_models/nn_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/nn_model.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #predicting for the test data
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print("Test Accuracy of the Neural Network :: ",validation_acc)


''' This functions loads the pre trained model of nerual network and test and prints the accuracy for each 
for the testing data using tfidf'''
def test_nn_model_tfidf(data,class_list):

    json_file = open('best_models/nn_model_tfidf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/nn_model_tfidf.h5")
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


''' This functions loads the pre trained model of nerual network and test and prints the accuracy for each 
for the testing data using kcore'''
def test_nn_model_kcore(data,class_list):

    json_file = open('best_models/nn_model_kcore.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/nn_model_kcore.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data=transfrom_kcore(data)
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)

''' This functions loads the pre trained model of nerual network and test and prints the accuracy for each 
for the testing data using coloring based features'''
def test_nn_model_coloring(data,class_list):
    #loading the pretrained the model
    json_file = open('best_models/nn_model_coloring.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/nn_model_coloring.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    data=transfrom_coloring(data)
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''

    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)


''' This functions loads the pre trained model of convolutional nerual network and test and prints the accuracy for each 
for the testing data using word count based features'''
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
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)


''' This functions loads the pre trained model of convolutional nerual network and test and prints the accuracy for each 
for the testing data using kcore'''

def test_cnn_model_kcore(data,class_list):
    data=transfrom_kcore(data)
    data = np.expand_dims(data, axis=2)
    json_file = open('best_models/cnn_model_core.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/cnn_model_core.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)


''' This functions loads the pre trained model of convolutional nerual network and test and prints the accuracy for each 
for the testing data using coloring based features'''

def test_cnn_model_coloring(data,class_list):
    data=transfrom_coloring(data)
    data = np.expand_dims(data, axis=2)
    json_file = open('best_models/cnn_model_coloring.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/cnn_model_coloring.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
    for i in range(0, len(data)):
        if (predict[i] > 0.25):
            predict_out.append(1)
        else:
            predict_out.append(0)

    validation_acc = accuracy_score(class_list, predict_out)
    print(validation_acc)


''' This functions loads the pre trained model of convolutional nerual network and test and prints the accuracy for each 
for the testing data using tfidf'''
def test_cnn_model_tfidf(data,class_list):

    data = np.expand_dims(data, axis=2)
    json_file = open('best_models/cnn_model_tfidf.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("best_models/cnn_model_tfidf.h5")
    print("Loaded model from disk")
    loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    predict = loaded_model.predict(data)
    predict_out = []
    ''' classfying based on the threshold found that 0.25 is opt theshold for the 
     this task'''
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
    #test_logist_model(count_vec.toarray().astype(int), class_list)
    #test_random_forest_model(count_vec.toarray().astype(int), class_list)
    test_nn_model(count_vec.toarray().astype(int),class_list)
    #test_cnn_model(count_vec.toarray().astype(int), class_list)

    #test_logist_model_kcore(count_vec,class_list)
    #test_random_forest_kcore(count_vec,class_list)
    #test_nn_model_kcore(count_vec,class_list)
    #test_cnn_model_kcore(count_vec,class_list)

    #test_logist_model_coloring(count_vec,class_list)
    #test_random_forest_coloring(count_vec,class_list)
    #test_nn_model_coloring(count_vec,class_list)
    #test_cnn_model_coloring(count_vec,class_list)

    ## Tf-dff transformation
    #tfidf=transfrom_word_tfidf(count_vec)
    #test_logist_model_tfidf(tfidf.toarray().astype(int),class_list)
    #test_random_forest_tfidf(tfidf.toarray().astype(int),class_list)
    #test_nn_model_tfidf(tfidf.toarray().astype(int),class_list)
    #test_cnn_model_tfidf(tfidf.toarray().astype(int),class_list)