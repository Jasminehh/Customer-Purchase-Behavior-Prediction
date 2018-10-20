import pandas as pd
import numpy as np

from LogisticModel import LogisticModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def setup_data(df_train, df_test):
    '''
    INPUT:
    df_train, the dataframe used for training your model
    df_test, the dataframe used for testing your model
    OUPUT:
    This function is to drop unnecessary features and split the target out of the feature dataframe
    '''
    X_train = df_train.drop(['transactionRevenue', 'yrMon'],axis = 1)
    y_train = X_train.pop('isPurchase')
    X_test = df_test.drop(['transactionRevenue', 'yrMon'],axis = 1)
    y_test = X_test.pop('isPurchase')
    return X_train, y_train, X_test, y_test

def drop_feature(X_train, X_test, features):
    '''
    INPUT:
    X_train, the feature dataframe used for training your model
    X_test, the feature dataframe used for testing your model
    features, the list of the features with VIF > 5, or the unuseful features you would like to drop
    OUPUT:
    This function is to drop unnecessary features
    '''
    X_train = X_train.drop(features,axis = 1)
    X_test = X_test.drop(features,axis = 1)
    return X_train, X_test

def report_accuracy(y_test,y_predict):
    '''
    INPUT:
    y_test, the numpy array of your target
    y_predict, the numpy array of your predicted target based on your model
    OUPUT:
    This function prints a classification report for your model.
    '''
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_predict)
    precision = metrics.precision_score(y_true=y_test, y_pred=y_predict)
    recall = metrics.recall_score(y_true=y_test, y_pred=y_predict)
    classification_report= metrics.classification_report(y_true=y_test, y_pred=y_predict)

    print('accuracy: {:.3f}'.format(accuracy))
    print('precision: {:.3f}'.format(precision))
    print('recall: {:.3f}'.format(recall))
    print('=========================================================')
    print('classification_report: \n{}'.format(classification_report))
    return accuracy

def plot_pca(x_pca, y_train,figsize=(16,8)):
    '''
    INPUT:
    x_pca, the dataframe used for scatter plot
    y_train, the numpy array of your target to train your model
    figsize, the default figure size is 16*8
    OUPUT:
    This function plots two scatter plots to understand your features.
    '''
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].scatter(x_pca[:,0],x_pca[:,1],c=y_train,cmap='plasma', alpha=0.2)
    axes[0].set_xlim([-30, 500])
    axes[0].set_ylim([-200, 400])
    axes[0].set_title('Non-Purchace v.s. Purchased',fontsize=20)
    axes[0].set_xlabel('First Principal Component',fontsize=15)
    axes[0].set_ylabel('Second Principal Component',fontsize=15)

    x_pca_1 = x_pca[y_train == 1]
    axes[1].scatter(x_pca_1[:,0],x_pca_1[:,1],c='y',cmap='plasma', alpha=0.2)
    axes[1].set_xlim([-30, 500])
    axes[1].set_ylim([-200, 400])
    axes[1].set_title('Purchased',fontsize=20)
    axes[1].set_xlabel('First Principal Component',fontsize=15)
    axes[1].set_ylabel('Second Principal Component',fontsize=15)
    plt.show()

def plot_pca_heatmap(df_comp, filename):
    '''
    INPUT:
    df_comp, the dataframe of your two pinciple components
    filename, the filename that you used to save the heatmap
    OUPUT:
    This function plots a heatmap of your pinciple components.
    '''
    xIndex = ['Visit Number','Hits','New Visits','Browser','Operating System','Device','Source']
    yIndex = ['FPC','SPC']

    plt.figure(figsize=(12,4))
    heatmap_comp = sns.heatmap(df_comp,cmap='plasma')
    heatmap_comp.set_title("Principal Component Heatmap", fontsize = 20)
    heatmap_comp.set_xticklabels(xIndex,rotation=45,fontsize = 14)
    heatmap_comp.set_yticklabels(yIndex,fontsize=14)
    heatmap_comp.savefig('figures/'+filename+'.png', bbox_inches='tight')


def create_classifier_model(classifier_model, X_train, y_train, X_test, y_test):
    '''
    INPUT:
    classifier_model, the classifier models (including SVC, Decision Tree, and Random Forest) you would like to use.
    X_train, y_train, X_test, y_test: the dataframes you used to train and test your model
    OUPUT:
    This function pirnts the accuracy report of your model and return the model accuracy
    '''
    if classifier_model = 'SVC':
        model = SVC()
    elif classifier_model = 'Decision Tree':
        model = DecisionTreeClassifier()
    elif classifier_model = 'RandomForest':
        model = RandomForestClassifier(n_estimators=600)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    report_accuracy(y_test, y_predict)
    return model

if __name__ == '__main__':

    df_train = pd.read_csv('data/train_code.csv')
    df_test = pd.read_csv('data/test_code.csv')

    X_train, y_train, X_test, y_test = setup_data(df_train, df_test)
    feature_highVIF = ['pageviews','channelGrouping_code','continent_code']
    X_train, X_test = drop_feature(X_train, X_test, feature_highVIF)

    # Create Logistic Regression Model
    log_model = LogisticModel(X_train, y_train, X_test, y_test)
    y_predict = log_model.predict()
    report_accuracy(y_test, y_predict)

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_train)
    x_pca = pca.transform(X_train)
    plot_pca(x_pca, y_train)
    df_comp = pd.DataFrame(pca.components_,columns=X_train.columns)
    plot_pca_heatmap(df_comp, 'pca_heatmap')

    # K-Means Clustering
    # Cluster the visitors into 2: Purchased or Not purchased
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x_pca)

    # SVC, Decision Tree, Random Forest
    svc_model = create_classifier_model('SVC', X_train, y_train, X_test, y_test)
    dt_model = create_classifier_model('Decision Tree', X_train, y_train, X_test, y_test)
    rf_model = create_classifier_model('RandomForest', X_train, y_train, X_test, y_test)
