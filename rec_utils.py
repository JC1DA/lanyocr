from utils import *
from supported_chars import character

# ref: https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.4/ppocr


def normalize_img(img, imgH, imgW):
    h, w = img.shape[:2]

    if h != imgH or w != imgW:
        ratio = w / float(h)

        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
    else:
        resized_image = img

    resized_image = resized_image.astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5

    if h != imgH or w != imgW:
        final_img = np.zeros(shape=(3, imgH, imgW), dtype=np.float32)
        final_img[:, :, :resized_w] = resized_image
    else:
        final_img = resized_image

    return final_img


def get_ignored_tokens():
    return [0]  # for ctc blank


def decode(text_index, text_prob=None, is_remove_duplicate=False):
    """convert text-index into text-label."""
    result_list = []
    ignored_tokens = get_ignored_tokens()
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        char_list = []
        # char_indices = []
        conf_list = []
        for idx in range(len(text_index[batch_idx])):
            if text_index[batch_idx][idx] in ignored_tokens:
                continue
            if is_remove_duplicate:
                # only for predict
                if (
                    idx > 0
                    and text_index[batch_idx][idx - 1] == text_index[batch_idx][idx]
                ):
                    continue
            char_list.append(character[int(text_index[batch_idx][idx])])
            # char_indices.append(int(text_index[batch_idx][idx]))
            if text_prob is not None:
                conf_list.append(text_prob[batch_idx][idx])
            else:
                conf_list.append(1)

        text = "".join(char_list)
        prob = 1.0 if conf_list else 0
        for p in conf_list:
            prob *= p

        result_list.append((text, prob))
    return result_list


def recognize_text(recognizer_session, img, model_h=32, model_w=320):
    norm_img = normalize_img(img, model_h, model_w)
    # norm_img = norm_img.astype(np.float16)
    preds = recognizer_session.run(None, {"x": [norm_img]})[0][0]
    preds_idx = preds.argmax(axis=1)
    preds_prob = preds.max(axis=1)
    results = decode([preds_idx], [preds_prob], True)
    text, prob = results[0]
    return text, prob


def recognize_text_batch(recognizer_session, imgs, model_h=32, model_w=320):
    batch_results = []
    normed_imgs = [normalize_img(img, model_h, model_w) for img in imgs]
    batch_preds = recognizer_session.run(None, {"x": normed_imgs})[0]

    for preds in batch_preds:
        preds_idx = preds.argmax(axis=1)
        preds_prob = preds.max(axis=1)
        results = decode([preds_idx], [preds_prob], True)
        batch_results.append(results[0])

    return batch_results
