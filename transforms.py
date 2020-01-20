import dicom_numpy
import numpy as np
import scipy
import skimage.transform
import torch
from numba import jit

class ToTensor(object):
    """Convert pydicom image in sample to Tensors."""

    def __init__(self, spacing=None, num_slices=None, aspect='sagittal', cut=(slice(None,None,1),slice(None,None,1))):
        if spacing is not None:
            assert isinstance(spacing, (list, tuple))
        if num_slices is not None:
            assert isinstance(num_slices, (int))
        self.spacing = spacing
        self.num_slices = num_slices
        self.aspect = aspect
        self.cut = cut

    #@jit(nopython=True, parallel=True)
    def __call__(self, sample):
        # Las imágenes son de un único color en tonos.  Al color se le suele llamar canal.
        # Así, deberíamos devolver un tensor de este tipo:  C X S X H X W,
        # donde C es canal y es igual a 1, S es el número de "slices" (cortes), H es "height"
        # (la altura) y W es "width" (el ancho).

        # Obtenemos una lista de matrices numpy con los cortes, es decir, un cubo.
        try:
            image, ijk_to_xyz = dicom_numpy.combine_slices(sample)
            image = image.astype(np.float32)
        except Exception as e:
            # Invalid DICOM data
            # We go without help
            try:
                slices = sorted(sample, key=lambda s: s.SliceLocation)
                image = [s.pixel_array for s in slices]
                image = np.array(image).astype(np.float32)
                del slices
            except Exception as e:
                # Cruzamos los dedos
                image = [s.pixel_array for s in sample]
                image = np.array(image).astype(np.float32)

        if self.spacing is not None:
            # Nuevo tamaño de los voxel
            new_spacing = self.spacing

            try:
                # Tamaño actual de los voxel
                spacing = map(float, ([sample[0].SliceThickness] + list(sample[0].PixelSpacing)))
                spacing = np.array(list(spacing))
            except Exception as e:
                global la_imagen
                print(sample[0][0x00180050])
                la_imagen = sample[0]
                raise e

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
        if self.num_slices is not None:            
            (deep, height, width) = image.shape

            if self.num_slices != deep:
                real_resize_factor = np.array([self.num_slices, height, width]) / image.shape

                # Reconstrucción de la imagen al nuevo tamaño de voxel
                image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, order=1)

        # Un canal:
        if self.aspect == 'sagittal':
            channel = [image[self.cut[0], self.cut[1], self.cut[2]]]
        elif self.aspect == 'axial':
            image = np.rot90(image)
            image = np.rot90(image, k=-1, axes=(1,2))
            channel = [image[self.cut[0], self.cut[1], self.cut[2]]]
        elif self.aspect == 'coronal':
            pass

        return torch.from_numpy(np.array(channel))
