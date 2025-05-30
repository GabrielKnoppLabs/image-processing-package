from skimage.io import imread, imsave
import os
import numpy as np


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_image(path: str, is_gray: bool = False) -> np.ndarray:
    """
    Lê uma imagem do disco.

    Args:
        path (str): Caminho do arquivo de imagem.
        is_gray (bool): Se True, carrega a imagem em escala de cinza.

    Returns:
        np.ndarray: Imagem carregada.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Extensão inválida '{ext}'. Extensões aceitas: {', '.join(VALID_EXTENSIONS)}")

    image = imread(path, as_gray=is_gray)
    return image


def save_image(image: np.ndarray, path: str) -> str:
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

    ext = os.path.splitext(path)[1].lower()
    if ext not in VALID_EXTENSIONS:
        raise ValueError(f"Extensão inválida '{ext}'. Use uma das extensões suportadas: {', '.join(VALID_EXTENSIONS)}")

    # Converte imagem para o intervalo [0, 255] se necessário
    if image.dtype in [np.float32, np.float64]:
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

    imsave(path, image)
    return path
