from pathlib import Path
import csv
import numpy as np

def duplicate_classes(data_dir = '', classes = [], save_labels=False):
    saved_files = 0
    for _class in classes:
        #
        # Obtiene el listado de todos los archivos *.np en el directorio de la clase.
        #
        files = list(Path(data_dir + '/' + _class).rglob("*.np"))

        for file in files:

            with file.open(mode='rb') as data:
                # Carga los datos numpy
                [image, label] = np.load(data, allow_pickle=True)
                data.close()

                # Invierte los planos
                image_reverse = np.array([image[0][::-1]])

                # Chequeo
                assert(np.array_equal(image_reverse[0][0], image[0][-1]))

                # Guarda el nuevo archivo
                filename = data_dir + '/' + _class + '/' + file.stem + 'r' + '.np'
                with open(filename, 'wb') as file_reversed:
                    print('Saving {}...'.format(filename))
                    np.save(file_reversed, np.array([image_reverse, label], dtype=object))
                    saved_files += 1
                    file_reversed.close()

                if save_labels:
                    # Se guarda la etiqueta del nuevo archivo en el archivo de etiquetas
                    id = file.stem[1::] + 'r'
                    with open('labels.csv', 'a', newline='') as labels_file:
                        writer = csv.writer(labels_file)
                        writer.writerow([id,_class])
                        labels_file.close()


def main():
    duplicate_classes('./NumpyADNI', ['AD','LMCI'], False)

# Si corre como programa principal y no como módulo:
if __name__ == '__main__':
    main()