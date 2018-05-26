from sklearn.model_selection import train_test_split
import glob
'''
This function creates train and validation sets to build model.
Also test set to test the model.
'''
def create_train_validation_test_sets(input_dir):
    txtfiles=[]
    ''' Reading all files from the directory'''
    class_label=[]
    for filename in glob.glob(input_dir+"*.txt"):
        txtfiles.append(filename)
        '''labeling the files with file name : ham is considered as non-spam
        spam is considered as spam.
        '''
        if(filename.__contains__('ham')):
            class_label.append(1)
        else:
            class_label.append(0)
    '''spliting the data into train and test sets'''
    X_train, X_test, y_train, y_test = train_test_split(txtfiles, class_label, test_size = 0.1, random_state = 42)
    

if __name__ == "__main__":
    create_train_validation_test_sets("raw_input/")