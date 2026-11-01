import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_ppm(filepath):
    if not os.path.exists(filepath):
        print(f"Ошибка: Файл '{filepath}' не найден!")
        return None

    stream = open(filepath, "rb")
    bytes_data = bytearray(stream.read())
    stream.close()

    np_array = np.asarray(bytes_data, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Ошибка: Не удалось декодировать PPM файл '{filepath}'")
        return None

    return img


def nearest_neighbor_interp(img, x, y):
    h, w = img.shape[:2]
    rx = np.round(x).astype(int)
    ry = np.round(y).astype(int)

    mask = (rx >= 0) & (rx < w) & (ry >= 0) & (ry < h)
    out = np.zeros((*x.shape, 3) if img.ndim == 3 else x.shape, dtype=img.dtype)
    out[mask] = img[ry[mask], rx[mask]]
    return out


def bilinear_interp(img, x, y):
    h, w = img.shape[:2]
    l = np.floor(x).astype(int)
    k = np.floor(y).astype(int)
    a, b = x - l, y - k

    out = np.zeros((*x.shape, 3) if img.ndim == 3 else x.shape, dtype=np.float32)

    for dx in (0, 1):
        for dy in (0, 1):
            px = np.clip(l + dx, 0, w - 1)
            py = np.clip(k + dy, 0, h - 1)
            w_coeff = ((1 - a) if dx == 0 else a) * ((1 - b) if dy == 0 else b)
            if img.ndim == 3:
                w_coeff = w_coeff[..., np.newaxis]
            out += w_coeff * img[py, px]

    mask = (l >= 0) & (l < w - 1) & (k >= 0) & (k < h - 1)
    if img.ndim == 3:
        mask = mask[..., np.newaxis]
    return np.where(mask, out, 0).clip(0, 255).astype(np.uint8)


def bicubic_interp(img, x, y):
    h, w = img.shape[:2]
    l = np.floor(x).astype(int)
    k = np.floor(y).astype(int)
    a, b = x - l, y - k

    c1 = (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 4.0
    c2 = -a * (a - 2) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 4.0
    c3 = b * (a - 1) * (a - 2) * (a + 1) * (b - 2) * (b + 1) / 4.0
    c4 = a * b * (a - 2) * (a + 1) * (b - 2) * (b + 1) / 4.0
    c5 = -a * (a - 1) * (a - 2) * (b - 1) * (b - 2) * (b + 1) / 12.0
    c6 = -b * (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b - 2) / 12.0
    c7 = a * b * (a - 1) * (a - 2) * (b - 2) * (b + 1) / 12.0
    c8 = a * b * (a - 2) * (a + 1) * (b - 1) * (b - 2) / 12.0
    c9 = a * (a - 1) * (a + 1) * (b - 1) * (b - 2) * (b + 1) / 12.0
    c10 = b * (a - 1) * (a - 2) * (a + 1) * (b - 1) * (b + 1) / 12.0
    c11 = a * b * (a - 1) * (a - 2) * (b - 1) * (b - 2) / 36.0
    c12 = -a * b * (a - 1) * (a + 1) * (b - 2) * (b + 1) / 12.0
    c13 = -a * b * (a - 2) * (a + 1) * (b - 1) * (b + 1) / 12.0
    c14 = -a * b * (a - 1) * (a + 1) * (b - 1) * (b - 2) / 36.0
    c15 = -a * b * (a - 1) * (a - 2) * (b - 1) * (b + 1) / 36.0
    c16 = a * b * (a - 1) * (a + 1) * (b - 1) * (b + 1) / 36.0

    coeffs = [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16]
    offsets = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0, -1), (-1, 0), (1, -1), (-1, 1),
        (0, 2), (2, 0), (-1, -1), (1, 2),
        (2, 1), (-1, 2), (2, -1), (2, 2)
    ]

    out = np.zeros((*x.shape, 3) if img.ndim == 3 else x.shape, dtype=np.float32)
    for coeff, (dx, dy) in zip(coeffs, offsets):
        px = np.clip(l + dx, 0, w - 1)
        py = np.clip(k + dy, 0, h - 1)
        c_weight = coeff[..., np.newaxis] if img.ndim == 3 else coeff
        out += c_weight * img[py, px]

    mask = (l >= 1) & (l < w - 2) & (k >= 1) & (k < h - 2)
    if img.ndim == 3:
        mask = mask[..., np.newaxis]
    return np.where(mask, out, 0).clip(0, 255).astype(np.uint8)


def transform(img, matrix, interp_fn, out_shape):
    h_out, w_out = out_shape
    grid_y, grid_x = np.indices((h_out, w_out))
    coords = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1)

    inv_mat = np.linalg.inv(matrix)
    src_coords = coords @ inv_mat.T
    return interp_fn(img, src_coords[..., 0], src_coords[..., 1])


image_path = "котик.ppm"
img = load_ppm(image_path)

if img is not None:
    h, w = img.shape[:2]

    M_scale = np.array([[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1]], dtype=np.float32)

    theta = np.radians(30)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = w / 2, h / 2
    M_rot = np.array([
        [cos_t, -sin_t, cx * (1 - cos_t) + cy * sin_t],
        [sin_t, cos_t, cy * (1 - cos_t) - cx * sin_t],
        [0, 0, 1]
    ], dtype=np.float32)

    M_shear = np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    transforms = [
        ("Масштабирование 1.5x", M_scale, (int(h * 1.5), int(w * 1.5))),
        ("Поворот на 30°", M_rot, (h, w)),
        ("Скос (Shear)", M_shear, (h, int(w * 1.3)))
    ]

    methods = [
        ("Ближайший сосед", nearest_neighbor_interp),
        ("Билинейная", bilinear_interp),
        ("Бикубическая", bicubic_interp)
    ]

    fig, axes = plt.subplots(len(transforms), 3, figsize=(11, 9))

    for i, (t_name, M, out_shape) in enumerate(transforms):
        for j, (m_name, interp_fn) in enumerate(methods):
            res = transform(img, M, interp_fn, out_shape)
            axes[i, j].imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{t_name}\n({m_name})", fontsize=8)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()