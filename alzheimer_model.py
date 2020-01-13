# encoding: utf-8

"""
The main implementation.
"""

N_CLASSES = 6
CLASS_NAMES = [ 'CN', 'SMC', 'EMCI', 'MCI', 'LMCI', 'AD']
DATA_DIR = './ADNI'
BATCH_SIZE = 5
EPOCHS = 1

from transforms import ToTensor
from adni_dataset import ADNI_Dataset

from models.densenet import densenet121

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import numpy as np

def test():
    # Sets device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional about GPU
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    # Optimiza la corrida
    cudnn.benchmark = True

    # Transformaciones de cada resonancia de magnética
    transform = transforms.Compose([ToTensor(spacing=[1,1,1], num_slices=256, aspect='axial', cut=(slice(100,200,1),slice(60,220,1),slice(90,200,1)))]) # Hace falta normalizar pero la función de pytorch no funciona en cubos

    # Conjunto de datos con las transformaciones especificadas anteriormente
    adni_dataset = ADNI_Dataset(data_dir=DATA_DIR, transform=transform)

    # Entrenamiento y prueba
    train_size = int(0.7 * len(adni_dataset))
    test_size = len(adni_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(adni_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1)

    # Inicializa y carga el modelo
    model = densenet121(channels=1, num_classes=len(CLASS_NAMES)).cuda()
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    # Imprime el modelo:
    #summary(model, adni_dataset[0][0].shape)

    # Función de pérdida:
    # Es la función usada para evaluar una solución candidata, es decir, la topología diseñada con sus pesos.
    criterion = nn.CrossEntropyLoss() # Entropía cruzada

    # Optimizador:
    # Estas son optimizaciones al algoritmo de descenso por gradiente para evitar mínimos locales en la búsqueda.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5) # SGD: Descenso por gradiente estocástico

    # Ciclo de entrenamiento:
    for epoch in range(EPOCHS):
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
            if i % 10 == 9:    # print every 10 batches
                print('[epoch %d, batch %5d] pérdida: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

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
    plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES)

    # Imprime estadísticas
    print('Accuracy: ', accuracy_score(test, predicted))
    print('Specificity (precision)', precision_score(test, predicted, average='macro'))
    print('Sensitivity (recall)', recall_score(test, predicted, average='macro'))

    # ROC curve
    plot_ROC_curve(test, torch.sigmoid(torch.Tensor(predicted)).numpy())
    print("Area Under ROC Curve (AUROC): {:.3f}".format(roc_auc_score(test, torch.sigmoid(torch.Tensor(predicted)).numpy())))


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


def plot_ROC_curve(true, predicted, color = "red", label = None):
    fp_r, tp_r, treshold = roc_curve(true, predicted)
    plt.plot(fp_r, tp_r, lw = 1, color = color, label = label)
    plt.plot([0, 1], [0, 1], lw = 1, color = "black")
    plt.xlabel("Rate of false positives")
    plt.ylabel("Rate of false negatives")
    plt.title("ROC curve")


# Si corre como programa principal y no como módulo:
if __name__ == '__main__':
    test()
