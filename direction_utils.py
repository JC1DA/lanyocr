import cv2
import numpy as np


def resize_norm_img(img, image_shape, padding=True):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_w = imgW
    else:
        ratio = imgH / float(h)
        _w = int(ratio * w)

        _img = cv2.resize(img, (_w, imgH))

        resized_image = np.zeros(shape=(imgH, imgW, 3), dtype=np.uint8)
        if _w > imgW:
            resized_image[:, :, :] = _img[:, :imgW, :]
        else:
            resized_image[:, :_w, :] = _img[:, :, :]

    resized_image = resized_image.astype("float32")
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5

    return resized_image


def is_flipped(direction_classifier_session, img, thresh=0.9):
    normed_img = resize_norm_img(img, [3, 48, 192])
    rotated_probs = direction_classifier_session.run(None, {"x": [normed_img]})[0][0]
    return rotated_probs[1] > thresh


def is_flipped_batch(direction_classifier_session, imgs, thresh=0.9):
    normed_imgs = []
    for img in imgs:
        normed_img = resize_norm_img(img, [3, 48, 192])
        normed_imgs.append(normed_img)

    rotated_probs = direction_classifier_session.run(None, {"x": normed_imgs})[0]
    return [True if p[1] > thresh else False for p in rotated_probs]


def flip_image_if_needed(direction_classifier_session, img, thresh=0.95):
    if is_flipped(direction_classifier_session, img, thresh):
        img = cv2.rotate(img, cv2.cv2.ROTATE_180)

    return img
