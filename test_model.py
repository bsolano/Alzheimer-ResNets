# encoding: utf-8

"""
The main implementation.
"""
import sys

from lib.functions import *
from models.densenet import densenet121
from adni_dataset import NumpyADNI_Dataset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import SubsetRandomSampler

def test(file, class_names, data_dir, results_dir):
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

    # Conjunto de datos con las transformaciones especificadas anteriormente
    adni_dataset = NumpyADNI_Dataset(data_dir=data_dir)

    # Entrenamiento y prueba
    train_size = int(0.75 * len(adni_dataset))
    test_size = len(adni_dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(adni_dataset, [train_size, test_size])

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    print('%d MRI images in testing loader...' % (test_size))

    # Inicializa, carga y corre el modelo
    model = densenet121(channels=1, num_classes=len(class_names), drop_rate=0.7).cuda()
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(results_dir+'/'+file))
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
    print_info_and_plots(test, predicted, class_names)


# Si corre como programa principal y no como módulo:
if __name__ == '__main__':

    test(file=sys.argv[1], class_names=sys.argv[1], data_dir=sys.argv[1], results_dir=sys.argv[1])
