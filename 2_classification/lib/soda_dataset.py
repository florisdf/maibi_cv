import cv2
import numpy as np
from pathlib import Path


def get_gallery_and_queries_for_class(class_name):
    """
    Return the gallery image and query images for a certain class.
    """
    # All images from the give class
    img_paths = [str(img_path) for img_path in Path('data/sodas/').glob(f'{class_name}/*.jpg')]
    
    # We use the first image as a reference
    gallery_img_path = img_paths.pop(0)
    gallery_img = cv2.imread(gallery_img_path)[..., ::-1]

    # The other images are used as queries
    query_imgs = [
        cv2.imread(img_path)[..., ::-1]
        for img_path in img_paths
    ]

    return gallery_img, query_imgs


def get_gallery_and_queries():
    gallery_labels = [
        'CocaCola2L',
        'Pepsi2L',
        'Fanta2L',
        'Sprite2L'
    ]
    
    gallery = []
    queries = []
    query_labels = []

    for label in gallery_labels:
        g_img, q_imgs = get_gallery_and_queries_for_class(label)
        gallery.append(g_img)
        queries.extend(q_imgs)
        query_labels.extend([label] * len(q_imgs))
    
    return gallery, queries, np.array(gallery_labels), np.array(query_labels)