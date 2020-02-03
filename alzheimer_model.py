# encoding: utf-8

"""
The main implementation.
"""

CLASS_NAMES = ['CN','SMC','EMCI','MCI','LMCI','AD']
N_CLASSES = len(CLASS_NAMES)
DATA_DIR = './ADNI'
DATA_DIR = './NumpyADNI'
BATCH_SIZE = 5
EPOCHS = 80
RESULTS_DIR = './results'

from transforms import ToTensor
from adni_dataset import ADNI_Dataset
from adni_dataset import NumpyADNI_Dataset

from models.densenet import densenet121

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np

def test():
    import platform; print(platform.platform())
    import sys; print('Python ', sys.version)
    import pydicom; print('pydicom ', pydicom.__version__)
    
    # Sets device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Additional about GPU
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    # Optimiza la corrida
    cudnn.benchmark = True

    # Transformaciones de cada resonancia de magnética
    #transform = transforms.Compose([ToTensor(spacing=[1,1,1], num_slices=256, aspect='sagittal', cut=(slice(40,214,2),slice(50,200,2),slice(40,240,2)), normalize=True)]) # Hace falta normalizar pero la función de pytorch no funciona en cubos

    # Conjunto de datos con las transformaciones especificadas anteriormente
    adni_dataset = NumpyADNI_Dataset(data_dir=DATA_DIR)

    # Entrenamiento y prueba
    train_size = int(0.75 * len(adni_dataset))
    test_size = len(adni_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(adni_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    print('%d MRI images in training loader...' % (train_size))
    print('%d MRI images in testing loader...' % (test_size))

    # Inicializa y carga el modelo
    model = densenet121(channels=1, num_classes=len(CLASS_NAMES), drop_rate=0.7).cuda()
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # Imprime el modelo:
    #summary(model, adni_dataset[0][0].shape)

    # Función de pérdida:
    # Es la función usada para evaluar una solución candidata, es decir, la topología diseñada con sus pesos.
    criterion = nn.CrossEntropyLoss() # Entropía cruzada

    # Optimizador:
    # Estas son optimizaciones al algoritmo de descenso por gradiente para evitar mínimos locales en la búsqueda.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9) # SGD: Descenso por gradiente estocástico

    # Ciclo de entrenamiento:
    losses = []
    for epoch in range(EPOCHS):
        lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epochs=[59,79])
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)

            # Para no acumular gradientes
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            
        print('[epoch %d] pérdida: %.3f' % (epoch + 1, running_loss / train_size))
        losses.append([epoch + 1, running_loss / train_size])
        if epoch % 10 == 0:
            torch.save(model.state_dict(), RESULTS_DIR+'/'+device.type+'-epoch-'+str(epoch)+'-alzheimer-densenet121.pth')
        
    torch.save(model.state_dict(), RESULTS_DIR+'/'+device.type+'-alzheimer-densenet121.pth')

    model.eval()
    test = []
    predicted = []
    with torch.no_grad():
        for data in test_loader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)
            _, label = torch.max(labels, 1)
            test.append(label)

            outputs = model(inputs)

            _, predicted_value = torch.max(outputs.data, 1)
            predicted.append(predicted_value)

    # Imprime la matriz de confusión
    test = [x.item() for x in test]
    predicted = [x.item() for x in predicted]
    cnf_matrix = confusion_matrix(test, predicted)
    plot_confusion_matrix(cnf_matrix, title='', classes=CLASS_NAMES)

    # Imprime estadísticas
    print(classification_report(test, predicted, labels=list(range(N_CLASSES)), target_names=CLASS_NAMES))

    # loss plot
    plot_loss(losses)

    # ROC curve
    plot_ROC_curve(test, predicted, classes=CLASS_NAMES)


def lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epochs=[]):
    """Decay learning rate by lr_decay on predefined epochs"""
    if epoch not in lr_decay_epochs:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    return optimizer


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
    plt.figure(figsize=(7,7))
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

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
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
    if not title:
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


# Si corre como programa principal y no como módulo:
if __name__ == '__main__':
    test()
