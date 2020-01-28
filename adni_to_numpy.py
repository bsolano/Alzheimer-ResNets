import pydicom as dcm
from pathlib import Path
import itertools
import re
import csv
import torch
import numpy as np
import scipy
import skimage.transform
import os

CLASS_NAMES = [ 'CN', 'SMC', 'EMCI', 'MCI', 'LMCI', 'AD']
DATA_DIR = './ADNI'
SAVE_DIR = './NumpyADNI'

def getlabels():
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

def transform(sample, spacing=None, num_slices=None, aspect='sagittal', cut=(slice(None,None,1),slice(None,None,1),slice(None,None,1)), normalize=True):
    # Obtenemos una lista de matrices numpy con los cortes, es decir, un cubo.
    try:
        slices = sorted(sample, key=lambda s: s.SliceLocation)
        image = [s.pixel_array for s in slices]
        image = np.array(image).astype(np.float32)
        del slices
    except Exception as e:
        if len(sample) == 1:
            # Tal vez es Phillips en cuyo caso todos los cortes están en el mismo archivo
            image = sample[0].pixel_array
            image = np.array(image).astype(np.float32)
            assert len(image) > 1, 'There are no slices'
        else:
            # Cruzamos los dedos
            image = [s.pixel_array for s in sample]
            image = np.array(image).astype(np.float32)

    if spacing is not None:
        # Nuevo tamaño de los voxel
        new_spacing = spacing

        try:
            # Tamaño actual de los voxel
            spacing = map(float, ([sample[0].SliceThickness] + list(sample[0].PixelSpacing)))
            spacing = np.array(list(spacing))
        except Exception as e:
            # Tal vez es Phillips
            spacing = map(float, ([sample[0][0x52009230][0][0x00289110][0][0x00180050].value] + list(sample[0][0x52009230][0][0x00289110][0][0x00280030].value)))
            spacing = np.array(list(spacing))

        # Si los espaciados son diferentes
        if spacing[0] != new_spacing[0] and spacing[1] != new_spacing[1] and spacing[2] != new_spacing[2]:
            # Cálculo de las nuevas proporciones
            resize_factor = spacing / new_spacing
            new_real_shape = image.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor

            # Reconstrucción de la imagen al nuevo tamaño de voxel
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)

    # Siempre convertimos cada corte a 256x256, si el corte no es cuadrado
    # lo rellenamos.
    (deep, height, width) = image.shape
    if height > width:
        w = (height-width)//2
        for i in range(w):
            image = np.insert(image, width, [0], axis=2)
        for i in range(w):
            image = np.insert(image, 0, [0], axis=2)
    if height < width:
        h = (width-height)//2
        for i in range(h):
            image = np.insert(image, width, [0], axis=1)
        for i in range(w):
            image = np.insert(image, 0, [0], axis=1)
    # Reconstrucción de la imagen al nuevo tamaño de voxel
    if image.shape[1] != 256 and image.shape[2] != 256:
        real_resize_factor = np.array([deep, 256, 256]) / image.shape
        image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)

    # Establecemos el número de cortes:
    if num_slices is not None:            
        (deep, height, width) = image.shape

        if num_slices != deep:
            real_resize_factor = np.array([num_slices, height, width]) / image.shape

            # Reconstrucción de la imagen al nuevo tamaño de voxel
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)

    # Un canal:
    if aspect == 'sagittal':
        image = image[cut[0], cut[1], cut[2]]
    elif aspect == 'axial':
        pass
    elif aspect == 'coronal':
        pass
        
    if normalize is not None:
        image = 2*image/255-1

    return image


def main():
    os.mkdir(SAVE_DIR)
    for dir in CLASS_NAMES:
        os.mkdir(SAVE_DIR+'/'+dir)

    labels = getlabels()

    #
    # Obtiene el listado de todos los archivos *.dcm en el directorio ADNI.
    #
    files = list(Path(DATA_DIR).rglob("*.[dD][cC][mM]"))

    #
    # Cada "imagen" de resonancia magnética es en realidad un grupo de imágenes que generan una
    # tridimensionalidad.
    #
    # Se agrupan las rutas de los archivos imágenes en un diccionario que tiene una resonancia
    # por llave.
    #
    image_files = {}

    for key, group in itertools.groupby(sorted(files), lambda i: re.search('I\d+', str(i)).group()):
        image_files[key] = sorted(list(group))
        print('Reading {}.'.format(key))

    image_names = list(image_files.keys())

    del files # Conserva memoria

    files = 0
    for index in range(len(image_names)):

        key = image_names[index]

        #
        # Lectura de la imagen DICOM.  La imagen es tridimensional, es un arreglo de cortes (planos).
        #
        file = image_files[key]
        image = []
        for file in image_files[key]:
            image.append(dcm.dcmread(file.as_posix()))
        if transform is not None:
            try:
                image = transform(image, spacing=[1,1,1], num_slices=256, cut=(slice(40,214,2),slice(50,200,2),slice(40,240,2)),normalize=True)
            except Exception as e:
                msg = "Error reading files: {}."
                raise Exception(msg.format(image_files[key]))
        else:
            # Obtenemos una lista de matrices numpy con los cortes, es decir, un cubo.
            try:
                slices = sorted(image, key=lambda i: i.SliceLocation)
                image = [s.pixel_array for s in slices]
                image = np.array(image).astype(np.float32)
                del slices
            except Exception as e:
                # Cruzamos los dedos
                image = [i.pixel_array for i in image]
                image = np.array(image).astype(np.float32)

        #
        # Es necesario armar un arreglo binario (0,1) para la etiqueta
        #
        label = []
        for c in CLASS_NAMES:
            if c == labels[key]:
                label.append(1)
            else:
                label.append(0)
        
        filename = SAVE_DIR+'/'+labels[key]+'/'+key+'.np'
        with open(filename, 'wb') as file:
            print('Saving {}...'.format(filename))
            np.save(file, np.array([np.array([image]), np.array(label)]))
            files += 1
            file.close()
            
    print('Total files {}...'.format(files))

# Si corre como programa principal y no como módulo:
if __name__ == '__main__':
    main()