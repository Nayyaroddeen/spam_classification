from sklearn.model_selection import train_test_split
import glob
'''
This function creates train and validation sets to build model.
Also test set to test the model.
'''
def create_train_validation_test_sets(input_dir):
    txtfiles=[]

    for file in glob.glob(input_dir+"*.txt"):
        txtfiles.append(file)

if __name__ == "__main__":
    create_train_test("raw_input/")