import dicom_numpy
import numpy as np
import scipy
import skimage.transform
import torch

class ToTensor(object):
    """Convert pydicom image in sample to Tensors."""

    def __init__(self, spacing=None, num_slices=None, aspect='sagittal', cut=(slice(None,None,1),slice(None,None,1))):
        if spacing is not None:
            assert isinstance(spacing, (list, tuple))
        if num_slices is not None:
            assert isinstance(num_slices, (list, tuple))
        self.spacing = spacing
        self.num_slices = num_slices
        self.aspect = aspect
        self.cut = cut

    def __call__(self, sample):
        # Las imágenes son de un único color en tonos.  Al color se le suele llamar canal.
        # Así, deberíamos devolver un tensor de este tipo:  C X S X H X W,
        # donde C es canal y es igual a 1, S es el número de "slices" (cortes), H es "height"
        # (la altura) y W es "width" (el ancho).

        # Obtenemos una lista de matrices numpy con los cortes, es decir, un cubo.
        try:
            image, ijk_to_xyz = dicom_numpy.combine_slices(sample)
        except dicom_numpy.DicomImportException as e:
            # invalid DICOM data
            raise e

        if self.spacing is not None:
            # Nuevo tamaño de los voxel
            new_spacing = self.spacing
    
            # Tamaño actual de los voxel
            spacing = map(float, ([sample[0].SliceThickness] + list(sample[0].PixelSpacing)))
            spacing = np.array(list(spacing))
        
            # Cálculo de las nuevas proporciones
            resize_factor = spacing / new_spacing
            new_real_shape = image.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor
            
            # Reconstrucción de la imagen al nuevo tamaño de voxel
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

        # Siempre convertimos cada corte a 256x256, si el corte no es cuadrado
        # lo rellenamos.
        for slice in range(image.shape[0]):
            (height, width) = image[slice].shape
            if height > width:
                w = (height-width)//2
                for i in range(w):
                    image[slice] = np.insert(image[slice], width, [0], axis=1)
                for i in range(w):
                    image[slice] = np.insert(image[slice], 0, [0], axis=1)
            if height < width:
                h = (width-height)//2
                for i in range(h):
                    image[slice] = np.insert(image[slice], width, [0], axis=0)
                for i in range(w):
                    image[slice] = np.insert(image[slice], 0, [0], axis=0)
            image[slice] = skimage.transform.resize(image[slice], [256, 256])

        # Establecemos el número de planos:
        if self.num_slices is not None:
            if spacing and new_spacing:
                spacing = new_spacing
            else:
                # Tamaño actual de los voxel
                spacing = map(float, ([sample[0].SliceThickness] + sample[0].PixelSpacing))
                spacing = np.array(list(spacing))
                
            new_spacing = spacing
            new_spacing[0] = self.num_slices/image.shape[0]

            # Cálculo de las nuevas proporciones
            resize_factor = spacing / new_spacing
            new_real_shape = image.shape * resize_factor
            new_shape = np.round(new_real_shape)
            real_resize_factor = new_shape / image.shape
            new_spacing = spacing / real_resize_factor
            
            # Reconstrucción de la imagen al nuevo tamaño de voxel
            image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

        # Un canal:
        if self.aspect == 'sagittal':
            channel = [image[self.cut[0]:self.cut[1]:]]
        elif self.aspect == 'axial':
            image = np.rot90(image)
            image = np.rot90(image, k=-1, axes=(1,2))
            channel = [image[self.cut[0]:self.cut[1]:]]
        elif self.aspect == 'coronal':
            pass


        return torch.from_numpy(np.array(channel))
