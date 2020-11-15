from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np

import torch

def print_info_and_plots(test, predicted, class_names, losses=None):
    cnf_matrix = confusion_matrix(test, predicted)
    plot_confusion_matrix(cnf_matrix, title='', classes=class_names)

    # Imprime estadísticas
    print(classification_report(test, predicted, labels=list(range(len(class_names))), target_names=class_names))
    print('Accuracy: {:.2f}'.format(accuracy_score(test, predicted)))
    print('Specificity (precision) - micro: {:.2f}'.format(precision_score(test, predicted, average='micro')))
    print('Specificity (precision) - macro: {:.2f}'.format(precision_score(test, predicted, average='macro')))
    print('Sensitivity (recall) - micro: {:.2f}'.format(recall_score(test, predicted, average='micro')))
    print('Sensitivity (recall) - macro: {:.2f}'.format(recall_score(test, predicted, average='macro')))

    # loss plot
    if (losses is not None):
        plot_loss(losses)

    # ROC curve
    plot_ROC_curve(test, predicted, classes=class_names)


def plot_loss(losses):
    x = [losses[i][0] for i in range(len(losses))]
    y = [losses[i][1] for i in range(len(losses))]

    plt.plot(x, y, 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.close()


def plot_ROC_curve(true, predicted, classes):
    """
    Uses code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    """
    
    # Data
    true = label_binarize(true, classes=list(range(len(classes))))    
    predicted = torch.sigmoid(torch.Tensor(label_binarize(predicted, classes=list(range(len(classes)))))).numpy()
    n_classes = true.shape[1]

    # Line width
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(true.ravel(), predicted.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(6,6))
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='C0', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='C1', linestyle=':', linewidth=4)

    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color='C'+str(i+2), lw=lw,
                 label='ROC curve of {0} (area = {1:0.2f})'
                 ''.format(classes[i] if classes is not None else i, roc_auc[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    From:  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if title is None:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    plt.close()


def get_class_distribution(dataset_obj):
    count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
    
    for target in dataset_obj.targets:
        count_dict[dataset_obj.classes[target]] += 1
            
    return count_dict