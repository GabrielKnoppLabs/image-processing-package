from skimage.transform import resize


def resize_image(image, proportion):
    """
    Redimensiona uma imagem com base em uma proporção.

    Args:
        image (np.ndarray): Imagem de entrada (RGB ou escala de cinza).
        proportion (float): Valor entre 0 e 1 que define a nova proporção da imagem.

    Returns:
        np.ndarray: Imagem redimensionada.
    """
    if not (0 < proportion <= 1):
        raise ValueError("A proporção deve estar entre 0 (exclusive) e 1 (inclusive).")

    original_height, original_width = image.shape[:2]
    new_height = round(original_height * proportion)
    new_width = round(original_width * proportion)

    resized = resize(
        image,
        (new_height, new_width),
        anti_aliasing=True,
        preserve_range=True
    )

    return resized.astype(image.dtype)
