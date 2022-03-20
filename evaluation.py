import matplotlib.pyplot as plt
import pandas as pd

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