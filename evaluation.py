import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

def plot_predictions(preds, y_test, save, name):
    plt.figure()
    assert len(preds)==len(y_test)
    plt.plot(list(y_test), label='label')
    plt.plot(list(preds), label='predictions')
    plt.legend()
    plt.ylabel('Price')
    if(save):
        plt.savefig(name)
    else:
        plt.show()

def get_accuracy(preds, y_test):
    assert len(preds)==len(y_test)
    preds_arr=list(preds)
    y_test_arr=list(y_test)
    correct=0
    for i in range(1,len(y_test)):
        if((y_test_arr[i]-y_test_arr[i-1])<0 and (preds_arr[i]-preds_arr[i-1])<0):
            correct+=1
        elif((y_test_arr[i]-y_test_arr[i-1])>0 and (preds_arr[i]-preds_arr[i-1])>0):
            correct+=1
    return correct/(len(preds)-1)

#def direction_loss(y_true, y_predictions):
#    preds_arr=list(y_predictions)
#    y_test_arr=list(y_true)
#    correct=0
#    for i in range(1,len(y_true)):
#        if((y_test_arr[i]-y_test_arr[i-1])<0 and (preds_arr[i]-y_test_arr[i-1])<0):
#            correct+=1
#        elif((y_test_arr[i]-y_test_arr[i-1])>0 and (preds_arr[i]-y_test_arr[i-1])>0):
#            correct+=1
#    return 1-(correct/(len(y_predictions)-1))

def direction_loss(y_true, pred):
    print(np.shape(y_true))
    future_y=tf.roll(y_true,-1, axis=0)
    label_diff=y_true.__sub__(future_y)
    pred_diff=pred.__sub__(future_y)
    label_diff=tf.math.greater(label_diff, 0)
    pred_diff=tf.math.greater(pred_diff, 0)
    print(label_diff, pred_diff)
    xor_res=label_diff.__xor__(pred_diff)
    print(xor_res)
    print(int(tf.math.count_nonzero(xor_res)), int(tf.size(xor_res)))
    return int(tf.math.count_nonzero(xor_res))/int(tf.size(xor_res, out_type=tf.dtypes.int64))


def direction_acc(y_true, pred):
    y_true=list(y_true)
    pred=list(pred)
    correct=0
    for i in range(len(pred)):
        if(pred[i]>0 and y_true[i]>0):
            correct+=1
        elif(pred[i]<0 and y_true[i]<0):
            correct+=1
    return correct/len(pred)



def plot_predictions_diff(preds, close, save, name):
    plt.figure()
    assert len(preds)==len(close)
    final_preds=pd.Series()
    final_preds=pd.Series(list(close))+pd.Series(list(preds))
    plt.plot(list(close), label='label')
    plt.plot(list(final_preds), label='predictions')
    plt.legend()
    plt.ylabel('Price')
    if(save):
        plt.savefig(name)
    else:
        plt.show()


def get_accuracy_diff(preds, y_test, THRESH):
    assert len(preds)==len(y_test)
    preds_arr=list(preds)
    y_test_arr=list(y_test)
    correct=0
    for i in range(1,len(y_test)):
        if(y_test_arr[i]<THRESH and preds_arr[i]<THRESH):
            correct+=1
        elif(y_test_arr[i]>THRESH and preds_arr[i]>THRESH):
            correct+=1
    return correct/(len(preds)-1)


def get_correlation(preds, y_test):
    assert len(preds)==len(y_test)
    preds_arr=list(preds)
    y_test_arr=list(y_test)
    return pd.Series(preds_arr).corr(pd.Series(y_test_arr))