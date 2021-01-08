import os, pickle
from scipy.sparse import csr_matrix, load_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd

def make_char_array(charlist):
    charr=[]
    this_arr=charlist
    i=0
    while i<len(this_arr):
        if this_arr[i]!="^":
            charr.append(this_arr[i])
            i+=1
        else:
            charr[-1]+="".join(this_arr[i:i+2])
            i+=2
    return charr

def train_svm(data_path, commuting_matrix, c, Kernel, Degree, outpath):
    train_path=os.path.join(data_path, "final_train")
    
    A_train=load_npz(os.path.join(train_path, "A_mat.npz")).astype(np.int64)
    B_train=load_npz(os.path.join(train_path, "B_mat.npz")).astype(np.int64)
        
    train_app_data=pd.read_csv(os.path.join(train_path, "app_data.csv"))
    
    metapath_dict={
        "A":A_train,
        "B":B_train,
        "A^T":A_train.T,
        "B^T":B_train.T
    }
        
    char_arr=make_char_array(list(commuting_matrix))

    X_train=metapath_dict[char_arr[0]]
    y_train=np.array(train_app_data.app.apply(lambda x:1 if '.' in x else 0))
    
    for i in range(len(char_arr)):
        if i!=0:
            X_train=X_train*metapath_dict[char_arr[i]]    
    
    mdl = SVC(kernel=Kernel, C=c, degree=Degree).fit(X_train.todense(), y_train)

    if outpath is not None:
        try:
            print(os.getcwd())
            fn="svm_model_%s.pkl"%commuting_matrix.lower().replace("^","")
            fp=os.path.join(os.getcwd(),outpath, fn)
            print("Saving Model to %s"%fp)
            with open(fp, 'wb') as output:
                pickle.dump(mdl, output)
        except FileNotFoundError:
            fn="svm_model_%s.pkl"%commuting_matrix.lower().replace("^","")
            os.makedirs(outpath)
            fp=os.path.join(os.getcwd(),outpath, fn)
            print("Saving Model to %s"%fp)
            with open(fp, 'wb') as output:
                pickle.dump(mdl, output)

def train(data_path, commuting_matrices, c, Kernel, Degree, outpath=None):
    for matrix in commuting_matrices:
        print("Making model from commuting matrix %s"%matrix)
        train_svm(data_path, matrix, c, Kernel, Degree, outpath)
