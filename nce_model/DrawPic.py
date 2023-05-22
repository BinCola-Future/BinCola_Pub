# coding:utf-8
import imp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.set_loglevel("info") 
from sklearn import metrics
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

def heapMapPlot(data,key_list,title,logdir,sexi):
    colormap=plt.cm.RdBu
    data=np.array(data)
    fig,ax=plt.subplots(figsize=(17,17))
    sns.heatmap(pd.DataFrame(np.round(data,4),columns=key_list,index=key_list),annot=False,fmt='.2f',
                vmax=int(data.max())+1,vmin=int(data.min())-1,
                xticklabels=True,yticklabels=True,square=True,cmap=sexi)  #"YlGnBu"
    plt.savefig(os.path.join(logdir,title))

def DrawROC(test_y,test_pred,logdir):
    test_fpr, test_tpr, test_thresholds = metrics.roc_curve(test_y, test_pred, pos_label=1)

    data_dict = {'threshold': test_thresholds, 'tprs': test_tpr, 'fprs': test_fpr}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/best_test_roc.csv", index=False, sep=',')

    test_auc = metrics.auc(test_fpr, test_tpr)
    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(test_fpr, test_tpr, 'b',label='AUC = %0.2f' % test_auc)
    fig.savefig(str(logdir) + "/best_test_roc.pdf")
    plt.close(fig)
    return test_auc


def DrawRecall_Pre_F1(test_y,test_pred,logdir):
    precision, recall, thresholds = metrics.precision_recall_curve(test_y, test_pred,pos_label=1)

    data_dict = {'threshold': thresholds, 'precision': precision[0:-1], 'recall': recall[0:-1]}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/pre_recall.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('Precision-Recall')
    plt.plot(recall, precision, 'b')
    fig.savefig(str(logdir) + "/pre_recall.pdf")
    plt.close(fig)

    fig = plt.figure()
    plt.title('thresholds-TPR')
    plt.plot(thresholds, recall[0:-1], 'b')
    fig.savefig(str(logdir) + "/thresholds_tpr.pdf")
    plt.close(fig)

    f1_scores = []
    for i in range(len(precision)):
        f1_socre = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1_scores.append(f1_socre)

    data_dict = {'threshold': thresholds, 'f1_scores': f1_scores[0:-1]}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/thresholds_f1_score.csv", index=False, sep=',')

    fig = plt.figure()
    plt.title('thresholds_f1_score')
    plt.plot(thresholds, f1_scores[0:-1], 'b')
    fig.savefig(str(logdir) + "/thresholds_f1_score.pdf")
    plt.close(fig)

    DrawF1score_CDF(precision, recall, logdir)


def DrawF1score_CDF(precision,recall,logdir):
    f1_scores = []
    f1_scores_percents = []
    CDF_X = list(np.linspace(0, 1, num=100))  # f1-score-cdf的横坐标
    for i in range(len(precision)):
        f1_socre = (2*precision[i]*recall[i])/(precision[i]+recall[i])
        f1_scores.append(f1_socre)
    for CDF in CDF_X:
        f1_scores_percents.append(GetPercent_Of_F1_score(f1_scores,CDF))
    fig = plt.figure()
    plt.title('F1score-CDF')
    plt.plot(CDF_X, f1_scores_percents, 'b')
    fig.savefig(str(logdir) + "/F1score-CDF.pdf")
    plt.close(fig)

    data_dict = {'CDF_X': CDF_X, 'f1_scores_percents': f1_scores_percents}
    dataframe = pd.DataFrame(data_dict)
    dataframe.to_csv(str(logdir) + "/F1score-CDF.csv", index=False, sep=',')


def GetPercent_Of_F1_score(f1_scores,CDF):
    num = 0
    for f1_score in f1_scores:
        if f1_score <= CDF:
            num += 1
    percent = float(num)/len(f1_scores)
    return percent