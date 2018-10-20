import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics


class LogisticModel():
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        log_model = LogisticRegression()
        log_model = log_model.fit(self.X_train, self.y_train)
        self.log_model = log_model.fit(self.X_train, self.y_train)

    def get_model(self):
        return self.log_model

    def predict(self):
        y_predict = self.log_model.predict(self.X_test)
        return y_predict

    def compute_cnf_matrix(self):
        y_predict = self.log_model.predict(self.X_test)
        cnf_matrix = confusion_matrix(self.y_test, y_predict)
        return cnf_matrix

    def compute_tpr_fpr(self):
        '''
        INPUT: numpy array, numpy array
        OUTPUT: list, list, list
        Take a numpy array of the predicted probabilities and a numpy array of the
        true labels.
        Return the True Positive Rates, False Positive Rates and Thresholds for the
        ROC curve.
        '''
        probabilities = self.log_model.predict_proba(self.X_test)[:, 1]
        thresholds = np.sort(probabilities)

        tprs = []
        fprs = []

        num_positive_cases = sum(self.y_test)
        num_negative_cases = len(self.y_test) - num_positive_cases

        for threshold in thresholds:
            predicted_positive = probabilities >= threshold
            true_positives = np.sum(predicted_positive * self.y_test)
            false_positives = np.sum(predicted_positive) - true_positives
            tpr = true_positives / float(num_positive_cases)
            fpr = false_positives / float(num_negative_cases)

            fprs.append(fpr)
            tprs.append(tpr)

        return tprs, fprs

    def plot_confusion_matrix(cm, ax, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        """
        p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.set_title(title,fontsize=20)

        plt.colorbar(p)
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                     horizontalalignment="center", size = 20,
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        ax.set_ylabel('True label',fontsize=15)
        ax.set_xlabel('Predicted label',fontsize=15)

    def plot_roc_confusionMatrix(self):
        fig, axes = plt.subplots(1, 2, figsize=(15,6))
        axes[1].grid(False)

        y_predict = self.log_model.predict(self.X_test)
        cnf_matrix = confusion_matrix(self.y_test, y_predict)
        tpr, fpr = self.compute_tpr_fpr()

        axes[0].plot(fpr, tpr,label='ROC (area = %0.2f)' % metrics.auc(fpr,tpr),color='r')
        axes[0].plot([0, 1], [0, 1], linestyle='--', lw=2, color='b',label='Luck')
        axes[0].legend(loc="lower right")
        axes[0].set_xlabel("False Positive Rate",fontsize = 15)
        axes[0].set_ylabel("True Positive Rate",fontsize = 15)
        axes[0].set_title("Receiver Operating Characteristic",fontsize = 20)

        class_names = ["Non-Purchase","Purchase"]
        self.plot_confusion_matrix(cnf_matrix, axes[1], classes=class_names,normalize=True,title='Normalized Confusion Matrix')
