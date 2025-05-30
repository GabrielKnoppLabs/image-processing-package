from skimage.io import imread, imsave
import os


def read_image(path, is_gray=False):
    """
    Lê uma imagem do disco.

    Args:
        path (str): Caminho do arquivo de imagem.
        is_gray (bool): Se True, carrega a imagem em escala de cinza.

    Returns:
        np.ndarray: Imagem carregada.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    image = imread(path, as_gray=is_gray)
    return image


def save_image(image, path):
    """
    Salva uma imagem no disco.

    Args:
        image (np.ndarray): Imagem a ser salva.
        path (str): Caminho do arquivo de destino (com extensão).

    Returns:
        str: Caminho do arquivo salvo.
    """
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

    imsave(path, image)
    return path
