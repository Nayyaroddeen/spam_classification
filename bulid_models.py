from sklearn.feature_extraction.text import CountVectorizer
import glob
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, MaxPooling1D
from keras.models import Sequential

'''builds a word count model for the input data'''
def count_vector_model(list_of_docs):
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(list_of_docs)
    filename = 'count_vec_model.sav'
    pickle.dump(count_vect, open('best_models/'+filename, 'wb'))
    return train_counts

'''builds a tfidf  model for the input data'''
def tf_idf_model(list_of_docs):
    tfidf_transformer = TfidfVectorizer()
    train_tfidf = tfidf_transformer.fit_transform(list_of_docs)
    filename = 'tfidf_vec_model.sav'
    pickle.dump(tfidf_transformer, open('best_models/'+filename, 'wb'))
    return train_tfidf


''' takes file directory and train and validation files list and return
as list of strings and the corresponding class label'''
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

'''This function builds a logistic regression model using 5 fold cross 
validation and saves the best model of these five folds. '''
def Logist_Reg_K_Fold(data,class_list):
    seed = 7
    #intializing the five fold
    kfold = StratifiedKFold(n_splits=5
                            , shuffle=True, random_state=seed)
    base_acc=0
    #starting of the 5-fold cross validation
    for train, validation in kfold.split(data, class_list):
        train_data=data[train]
        train_class=class_list[train]

        validation_data=data[validation]
        validation_class=class_list[validation]

        logisticRegr = LogisticRegression()
        #logisticRegr.fit(train_data.toarray().astype(int), train_class)
        logisticRegr.fit(train_data, train_class)

        #x_test = logisticRegr.predict(validation_data.toarray().astype(int))
        x_test = logisticRegr.predict(validation_data)

        validation_acc=accuracy_score(validation_class, x_test)
        if(base_acc<validation_acc):
            filename = 'logistic_best_model.sav'
            pickle.dump(logisticRegr, open('best_models/' + filename, 'wb'))
            base_acc=validation_acc
        print(accuracy_score(validation_class, x_test))

'''This function builds a random forest  model using 5 fold cross 
validation and saves the best model of these five folds. '''

def Random_Forest_K_Fold(data,class_list):
    seed = 7
    # intializing the five fold
    kfold = StratifiedKFold(n_splits=5
                            , shuffle=True, random_state=seed)
    base_acc = 0
    #starting of the 5-fold cross validation
    for train, validation in kfold.split(data, class_list):
        train_data = data[train]
        train_class = class_list[train]

        validation_data = data[validation]
        validation_class = class_list[validation]
        print(train_data)
        clf = RandomForestClassifier(n_estimators=70)
        clf.fit(train_data,train_class)
        x_test = clf.predict(validation_data)
        validation_acc = accuracy_score(validation_class, x_test)
        if (base_acc < validation_acc):
            filename = 'random_forest_best_model.sav'
            pickle.dump(clf, open('best_models/' + filename, 'wb'))
            base_acc=validation_acc
        print(accuracy_score(validation_class, x_test))


'''This function builds a nerual network  model using 5 fold cross 
validation and saves the best model of these five folds. '''

def NN_K_Fold(data,class_list):
    seed = 7
    # intializing the five fold
    kfold = StratifiedKFold(n_splits=5
                            , shuffle=True, random_state=seed)
    base_acc = 0
    #starting of the 5-fold cross validation
    for train, validation in kfold.split(data, class_list):
        #selecting the training data for the fold
        train_data = data[train]
        train_class = class_list[train]
        #selecting the validation data for the fold
        validation_data = data[validation]
        validation_class = class_list[validation]
        #calculating the input dimention for the first layer of NN
        input_dim=len(train_data[1,:])
        hidden_nodes_len=1000
        print(input_dim)
        print(hidden_nodes_len)
        #building the three layered network
        model = Sequential()
        model.add(Dense(hidden_nodes_len, input_dim=input_dim, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(train_data, train_class, epochs=10, batch_size=50)


        scores = model.evaluate(train_data, train_class)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        predict = model.predict(validation_data)
        predict_out=[]
        #fixing the threshold .25 as it best for the experiment and predicting validation samples
        for i in range(0,len(validation_data)):
            if(predict[i]>0.25):
                predict_out.append(1)
            else:
                predict_out.append(0)

        validation_acc = accuracy_score(validation_class, predict_out)
        #saving the best model
        print(validation_acc)
        if (base_acc < validation_acc):
            filename = 'nn_best_model.sav'
            # serialize model to JSON
            model_json = model.to_json()
            with open("best_models/nn_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("best_models/nn_model.h5")
            print("Saved model to disk")
            base_acc=validation_acc
        print(accuracy_score(validation_class, predict_out))


'''This function builds a convolutional neural network  model using 5 fold cross 
validation and saves the best model of these five folds. '''

def CNN_K_Fold(data,class_list):
    seed = 7
    # intializing the five fold
    kfold = StratifiedKFold(n_splits=5
                            , shuffle=True, random_state=seed)
    base_acc = 0
    input_dim = len(data[1, :])
    hidden_nodes_len = 1000
    data = np.expand_dims(data, axis=2)
    #starting of the 5-fold cross validation
    for train, validation in kfold.split(data, class_list):
        train_data = data[train]
        train_class = class_list[train]

        validation_data = data[validation]
        validation_class = class_list[validation]


        print(input_dim)
        print(hidden_nodes_len)
        model = Sequential()
        model.add(Convolution1D(nb_filter=1,filter_length=100, padding="same", input_shape=(input_dim, 1)))
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=10, strides=10))
        model.add(Flatten())
        model.add(Dense(400, input_dim=input_dim, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.fit(train_data, train_class, epochs=10, batch_size=50)

        scores = model.evaluate(train_data, train_class)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        predict = model.predict(validation_data)
        predict_out=[]
        for i in range(0,len(validation_data)):
            if(predict[i]>0.25):
                predict_out.append(1)
            else:
                predict_out.append(0)

        validation_acc = accuracy_score(validation_class, predict_out)
        print(validation_acc)
        if (base_acc < validation_acc):
            filename = 'cnn_best_model.sav'
            # serialize model to JSON
            model_json = model.to_json()
            with open("best_models/cnn_model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights("best_models/cnn_model.h5")
            print("Saved model to disk")
            base_acc=validation_acc
        print(accuracy_score(validation_class, predict_out))

if __name__ == "__main__":
    class_list=[]
    a=[]
    train_validation_set=pickle.load(open('train_validation_sets.txt', 'rb'))
    count_vec,class_list=preprocess_docs('raw_input',train_validation_set)
    count_vec=count_vector_model(count_vec)
    #count_vec=tf_idf_model(count_vec)
    #Logist_Reg_K_Fold(count_vec, np.array(class_list))
    #Random_Forest_K_Fold(count_vec.toarray().astype(int), np.array(class_list))
    NN_K_Fold(count_vec.toarray().astype(int), np.array(class_list))
    #CNN_K_Fold(count_vec.toarray().astype(int), np.array(class_list))

