# Para la lectura de imágenes médicas
import pydicom as dcm
# Para la iteración en directorios
from pathlib import Path
# Para el agrupamiento
import itertools
# Para las expresiones regulares
import re
# Para manejar el archivo con las etiquetas
import csv
# Para el tensor
import torch
# Para manejar conjuntos de datos y heredar de la clase
from torch.utils.data import Dataset

CLASS_NAMES = [ 'CN', 'SMC', 'EMCI', 'MCI', 'LMCI', 'AD']

class ADNI_Dataset(Dataset):
    def __init__(self, data_dir='ADNI', transform=None):
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
            self.image_files[key] = list(group)
        
        self.image_names = list(self.image_files.keys())
        
        del files # Conserva memoria

        self.transform = transform        


    def __getlabels__(self):
        ''' '''
        labels = {}
        with open('labels.csv', 'r') as file:
            reader = csv.reader(file)
            index = 0
            for row in reader:
                if index != 0:
                    labels['I' + row[0]] = row[2]
                index += 1
        return labels


    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        
        key = self.image_names[index]

        #
        # Lectura de la imagen DICOM.  La imagen es tridimensional, es un arreglo de cortes (planos).
        #
        file = self.image_files[key]
        image = []
        for file in self.image_files[key]:
            image.append(dcm.dcmread(file.as_posix()))
        if self.transform is not None:
            image = self.transform(image)
        
        #
        # Es necesario armar un arreglo binario (0,1) para la etiqueta
        #
        labels = self.__getlabels__()
        label = []
        for c in CLASS_NAMES:
            if c == labels[key]:
                label.append(1)
            else:
                label.append(0)

        return image, torch.Tensor(label)

    def __len__(self):
        return len(self.image_names)