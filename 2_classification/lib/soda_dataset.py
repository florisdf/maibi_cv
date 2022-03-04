import cv2
import numpy as np
from pathlib import Path


DEFAULT_LABELS = [
    'CocaCola2L',
    'Pepsi2L',
    'Fanta2L',
    'Sprite2L'
]


def get_gallery_and_queries_for_class(class_name):
    """
    Return the gallery image and query images for a certain class.
    """
    gallery_img = cv2.imread(f'data/sodas/gallery/{class_name}/{class_name}_00.jpg')[..., ::-1]

    # The other images are used as queries
    query_imgs = [
        cv2.imread(str(img_path))[..., ::-1]
        for img_path in Path('data/sodas/queries').glob(f'{class_name}/*.jpg')
    ]

    return gallery_img, query_imgs


def get_gallery_and_queries(labels=DEFAULT_LABELS):
    
    gallery = []
    queries = []
    query_labels = []

    for label in labels:
        g_img, q_imgs = get_gallery_and_queries_for_class(label)
        gallery.append(g_img)
        queries.extend(q_imgs)
        query_labels.extend([label] * len(q_imgs))

    return gallery, queries, np.array(labels), np.array(query_labels)