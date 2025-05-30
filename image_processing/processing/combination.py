import numpy as np
from skimage.color import rgb2gray
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim


def find_difference(image1, image2, verbose=True):
    """
    Compara duas imagens e retorna uma imagem de diferença normalizada (0 a 1).

    Args:
        image1 (np.ndarray): Primeira imagem (RGB).
        image2 (np.ndarray): Segunda imagem (RGB).
        verbose (bool): Exibe a similaridade no terminal.

    Returns:
        np.ndarray: Mapa de diferença normalizado.
    """
    if image1.shape != image2.shape:
        raise ValueError("As imagens devem ter o mesmo formato (altura, largura, canais).")

    gray1 = rgb2gray(image1)
    gray2 = rgb2gray(image2)

    score, diff = ssim(gray1, gray2, full=True)

    if verbose:
        print(f"Similaridade entre as imagens: {score:.4f}")

    norm_diff = (diff - diff.min()) / (diff.max() - diff.min())
    return norm_diff


def transfer_histogram(source, reference):
    """
    Ajusta o histograma da imagem `source` para se parecer com o da imagem `reference`.

    Args:
        source (np.ndarray): Imagem que será ajustada.
        reference (np.ndarray): Imagem de referência para o histograma.

    Returns:
        np.ndarray: Nova imagem com histograma ajustado.
    """
    matched = match_histograms(source, reference, channel_axis=-1)
    return matched
