import cv2


def lightness_only(func):
    def wrapper(img, *args, **kwargs):
        if img.shape[-1] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = func(lab_planes[0], *args, **kwargs)
            lab = cv2.merge(lab_planes)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return func(img, *args, **kwargs)
    return wrapper