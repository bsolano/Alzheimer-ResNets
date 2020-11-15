import torch
import torchvision
import pydicom as dcm
import itertools
import re
import csv
import numpy as np
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class ADNI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='ADNI', class_names=['CN','SMC','EMCI','MCI','LMCI','AD'], transform=None):
        ''' '''

        #
        # Obtiene el listado de todos los archivos *.dcm en el directorio ADNI.
        #
        files = list(Path(data_dir).rglob("*.[dD][cC][mM]"))

        #
        # Cada "imagen" de resonancia magnética es en realidad un grupo de imágenes que generan una
        # tridimensionalidad.
        #
        # Se agrupan las rutas de los archivos imágenes en un diccionario que tiene una resonancia
        # por llave.
        #
        self.image_files = {}

        for key, group in itertools.groupby(sorted(files), lambda i: re.search('I\d+', str(i)).group()):
            self.image_files[key] = sorted(list(group))
        
        self.image_names = list(self.image_files.keys())
        self.transform = transform
        self.class_names = class_names
        self.labels = self.__getlabels__()

    def __getlabels__(self):
        ''' '''
        labels = {}
        with open('labels.csv', 'r') as file:
            reader = csv.reader(file)
            index = 0
            for row in reader:
                if index != 0:
                    labels['I' + row[0]] = row[1]
                index += 1
        return labels


    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its label
        """
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        key = self.image_names[index]

        #
        # Lectura de la imagen DICOM.  La imagen es tridimensional, es un arreglo de cortes (planos).
        #
        file = self.image_files[key]
        image = []
        for file in self.image_files[key]:
            image.append(dcm.dcmread(file.as_posix()))
        if self.transform is not None:
            try:
                image = self.transform(image)
            except Exception as e:
                msg = "Error reading files: {}."
                raise Exception(msg.format(self.image_files[key]))
        else:
            # Obtenemos una lista de matrices numpy con los cortes, es decir, un cubo.
            try:
                slices = sorted(image, key=lambda s: s.SliceLocation)
                image = [s.pixel_array for s in slices]
                image = np.array(image).astype(np.float32)
                del slices
            except Exception as e:
                if len(image) == 1:
                    # Tal vez es Phillips en cuyo caso todos los cortes están en el mismo archivo
                    image = image[0].pixel_array
                    image = np.array(image).astype(np.float32)
                    assert len(image) > 1, 'There are no slices'
                else:
                    # Cruzamos los dedos
                    image = [s.pixel_array for s in image]
                    image = np.array(image).astype(np.float32)
        
        #
        # Es necesario armar un arreglo binario (0,1) para la etiqueta
        #
        label = []
        for c in self.class_names:
            if c == self.labels[key]:
                label.append(1)
            else:
                label.append(0)

        return image, torch.LongTensor(label)

    def __len__(self):
        return len(self.image_names)


class NumpyADNI_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='NumpyADNI'):
        ''' '''

        #
        # Obtiene el listado de todos los archivos *.np en el directorio NumpyADNI.
        #
        self.image_files = list(Path(data_dir).rglob("*.np"))

        self.image_names = [file.name[:-3] for file in self.image_files]

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its label
        """
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        file = self.image_files[index]
        
        with file.open(mode='rb') as data:
            [image, label] = np.load(data, allow_pickle=True)
            data.close()

        return [torch.from_numpy(image), torch.from_numpy(label)]

    def __len__(self):
        return len(self.image_names)


def load_numpy_adni(path):
    with open(path, mode='rb') as data:
        [image, label] = np.load(data, allow_pickle=True)
        data.close()

    return torch.from_numpy(image), torch.from_numpy(np.array(label))


class NumpyADNI_FolderDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, data_dir='./NumpyADNI', class_names=['CN','EMCI','MCI','LMCI','AD']):
        ''' '''
        super(NumpyADNI_FolderDataset, self).__init__(root=data_dir, loader=load_numpy_adni, extensions=('.np'))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, class) where class is a vector of zeros and ones, with a one in the class.
        """
        path, target = self.samples[index]
        return self.loader(path)