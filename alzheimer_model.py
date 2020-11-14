# encoding: utf-8

"""
The main implementation.
"""

from transforms import ToTensor
from adni_dataset import ADNI_Dataset
from adni_dataset import NumpyADNI_Dataset
from lib.functions import *

from models.densenet import densenet121

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary

import pickle
import re

def test(class_names, data_dir, results_dir, epochs, batch_size, lr_decay_epochs=None, model_file=None, architecture='densenet121'):
    import platform; print(platform.platform())
    import sys; print('Python ', sys.version)
    import pydicom; print('pydicom ', pydicom.__version__)
    print('pytorch ', torch.__version__)
    
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
    adni_dataset = NumpyADNI_FolderDataset(data_dir=data_dir)

    # Sampler
    target_list = torch.tensor(adni_dataset.targets)
    target_list = target_list[torch.randperm(len(target_list))]
    class_count = [i for i in get_class_distribution(adni_dataset).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )

    # Entrenamiento y prueba
    train_size = int(0.75 * len(adni_dataset))
    test_size = len(adni_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(adni_dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=weighted_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    print('%d MRI images in training loader...' % (train_size))
    print('%d MRI images in testing loader...' % (test_size))

    # Inicializa y carga el modelo
    class_ = globals()[architecture]
    model = class_(channels=1, num_classes=len(class_names), drop_rate=0.7).to(device)
    model = torch.nn.DataParallel(model).to(device)
    if model_file is not None:
        model.load_state_dict(torch.load(results_dir+'/'+model_file))
        match = re.match( r'.*-epoch-(\d+)-.*', model_file, re.M)
        starting_epoch = int(match.group(1)) + 1
        print('Loaded file {}, restarting at epoch {}.'.format(model_file, starting_epoch))
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
    try: starting_epoch
    except NameError: starting_epoch = None
    if starting_epoch is None:
        epoch = 0
    else:
        epoch = starting_epoch
    while epoch < epochs:
        if lr_decay_epochs is not None:
            lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epochs=lr_decay_epochs)
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
            
        print('[epoch %d] pérdida: %.3f' % (epoch, running_loss / train_size))
        losses.append([epoch + 1, running_loss / train_size])
        with open(results_dir+'/'+device.type+'-epoch-'+str(epoch)+'-losses.dump', 'wb') as losses_file:
            pickle.dump(losses, losses_file)
            losses_file.close()
        if (epoch % 10 == 9) or (epoch >= 79):
            torch.save(model.state_dict(), results_dir+'/'+device.type+'-epoch-'+str(epoch)+'-alzheimer-' + architecture + '.pth')
        
        # Next epoch
        epoch += 1

    torch.save(model.state_dict(), results_dir+'/'+device.type+'-alzheimer-densenet121.pth')

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

    test = [x.item() for x in test]
    predicted = [x.item() for x in predicted]

    # Imprime estadísticas y gráficos
    print_info_and_plots(test, predicted, class_names, losses)


def lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epochs=[]):
    """Decay learning rate by lr_decay on predefined epochs"""
    if epoch not in lr_decay_epochs:
        return optimizer
    
    print('Learning rate decay of {} on epoch {}.'.format(lr_decay, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay

    return optimizer


# Si corre como programa principal y no como módulo:
if __name__ == '__main__':

    test(class_names=['CN','EMCI','MCI','LMCI','AD'],
         data_dir='./NumpyADNI',
         results_dir='./results',
         epochs=110,
         batch_size=5,
         lr_decay_epochs=[69])
