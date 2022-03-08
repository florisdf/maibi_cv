import cv2
import numpy as np
from pathlib import Path


DEFAULT_LABELS = [
    'CocaCola',
    'Pepsi',
    'Fanta',
    'Sprite'
]


def get_gallery_for_class(class_name):
    """
    Return the gallery image for a certain class.
    """
    gallery_img = cv2.imread(f'data/sodas/gallery/{class_name}/{class_name}_gallery_00.jpg')[..., ::-1]

    return gallery_img

def get_queries_for_class(class_name):
    """
    Return the query images for a certain class.
    """
    # The other images are used as queries
    query_imgs = [
        cv2.imread(str(img_path))[..., ::-1]
        for img_path in Path('data/sodas/query').glob(f'{class_name}/*.jpg')
    ]

    return query_imgs


def get_gallery_and_queries(labels=DEFAULT_LABELS):
    """
    Return the gallery and queries (along with their respective class labels) of the given labels.
    """
    gallery = []
    queries = []
    query_labels = []
    gallery_labels = []

    for label in labels:
        g_img = get_gallery_for_class(label)
        gallery.append(g_img)
        gallery_labels.append(label)

        q_imgs = get_queries_for_class(label)
        queries.extend(q_imgs)
        query_labels.extend([label] * len(q_imgs))

    return gallery, queries, np.array(gallery_labels), np.array(query_labels)