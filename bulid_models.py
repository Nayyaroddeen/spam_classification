from sklearn.feature_extraction.text import CountVectorizer
import glob
import pickle

def count_vector_model(list_of_docs):
    count_vect = CountVectorizer()
    train_counts = count_vect.fit_transform(list_of_docs)
    filename = 'count_vec_model.sav'
    pickle.dump(count_vect, open('best_models/'+filename, 'wb'))
    return train_counts

