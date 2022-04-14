from utils import *


def load_img(image_path):
    MAX_SQUARE_SIZE = 1536
    img = cv2.imread(image_path)

    img_resized, _, _ = resize_aspect_ratio(
        img, square_size=MAX_SQUARE_SIZE, interpolation=cv2.INTER_LINEAR
    )
    img_resized = img_resized[:, :, ::-1]
    img_resized = normalizeMeanVariance(img_resized)
    img_resized = np.transpose(img_resized, [2, 0, 1])
    return img_resized, img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


# ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py
def getDetBoxes_core(
    textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False
):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        mapper.append(k)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = (
            np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
            .transpose()
            .reshape(-1, 2)
        )
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)

    return det, labels, mapper


# ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py
def getDetBoxes(
    textmap,
    linkmap,
    text_threshold,
    link_threshold,
    low_text,
    poly=False,
    estimate_num_chars=False,
):
    if poly and estimate_num_chars:
        raise Exception(
            "Estimating the number of characters not currently supported with poly."
        )
    boxes, labels, mapper = getDetBoxes_core(
        textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars
    )

    if poly:
        # polys = getPoly_core(boxes, labels, mapper, linkmap)
        pass
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper


# ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/craft_utils.py
def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


# ref: https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/detection.py
def postprocess(
    y,
    original_img,
    max_img_size: int = 1536,
    text_threshold: float = 0.7,
    low_text_threshold: float = 0.33,
    # configs to detect more texts
    # text_threshold: float = 0.6,
    # low_text_threshold: float = 0.25,
    link_text_threshold: float = 0.125,
):
    height, width = original_img.shape[:2]

    target_size = max(height, width)
    if target_size > max_img_size:
        target_size = max_img_size

    target_ratio = float(target_size) / max(height, width)
    ratio_h = ratio_w = 1.0 / target_ratio

    score_text = y[:, :, 0]
    score_link = y[:, :, 1]

    # Post-processing
    estimate_num_chars = False
    boxes, polys, mapper = getDetBoxes(
        score_text,
        score_link,
        text_threshold=text_threshold,
        link_threshold=link_text_threshold,
        low_text=low_text_threshold,
        poly=False,
        estimate_num_chars=estimate_num_chars,
    )

    # coordinate adjustment
    boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = adjustResultCoordinates(polys, ratio_w, ratio_h)

    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    result = []
    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        result.append(poly)

    return result


def check_downward_mergable(
    text_line: LanyOcrTextLine,
    rrect: LanyOcrRRect,
    width_thresh=0.5,
    iou_thresh=0.01,
) -> bool:
    last_rrect_in_line = text_line.sub_rrects[-1]

    if not check_area_merge(last_rrect_in_line, rrect, thresh=0.75):
        return False

    min_hw = min(
        min(last_rrect_in_line.getWidth(), last_rrect_in_line.getHeight()),
        min(rrect.getWidth(), rrect.getHeight()),
    )

    # we are going downward, cur box has to be on the right of prev box
    if rrect.getCenterX() - last_rrect_in_line.getCenterX() < 0.5 * min_hw:
        return False

    if rrect.getCenterY() - last_rrect_in_line.getCenterY() < 0.25 * min_hw:
        return False

    if iou(rrect.getBoundingBox(), last_rrect_in_line.getBoundingBox()) < iou_thresh:
        return False

    max_last_x = last_rrect_in_line.maxX()
    min_cur_x = rrect.minX()

    if min_cur_x - max_last_x > min_hw * width_thresh:
        return False

    return True


def check_upward_mergable(
    text_line: LanyOcrTextLine,
    rrect: LanyOcrRRect,
    height_thresh=0.75,
    width_thresh=0.5,
    angle_thresh=20,
    iou_thresh=0.005,
) -> bool:
    last_rrect_in_line = text_line.sub_rrects[-1]

    if not check_area_merge(last_rrect_in_line, rrect):
        return False

    min_hw = min(
        min(last_rrect_in_line.getWidth(), last_rrect_in_line.getHeight()),
        min(rrect.getWidth(), rrect.getHeight()),
    )

    # we are going upward, cur box has to be on the right of prev box
    if rrect.getCenterX() - last_rrect_in_line.getCenterX() < 0.5 * min_hw:
        return False

    if last_rrect_in_line.getCenterY() - rrect.getCenterY() < 0.25 * min_hw:
        return False

    last_rrect_bbox = last_rrect_in_line.getBoundingBox()
    last_rrect_bbox.left *= 0.975
    last_rrect_bbox.top *= 0.975
    last_rrect_bbox.right *= 1.025
    last_rrect_bbox.bottom *= 1.025

    cur_rrect_bbox = rrect.getBoundingBox()
    cur_rrect_bbox.left *= 0.975
    cur_rrect_bbox.top *= 0.975
    cur_rrect_bbox.right *= 1.025
    cur_rrect_bbox.bottom *= 1.025

    _iou = iou(last_rrect_bbox, cur_rrect_bbox)
    if _iou < iou_thresh:
        return False

    max_last_x = last_rrect_in_line.maxX()
    min_cur_x = rrect.minX()

    if min_cur_x - max_last_x > min_hw * width_thresh:
        return False

    return True


def check_vertical_mergable(
    text_line: LanyOcrTextLine,
    rrect: LanyOcrRRect,
    iou_thresh=0.01,
) -> bool:
    last_rrect_in_line = text_line.sub_rrects[-1]

    if not check_area_merge(last_rrect_in_line, rrect, thresh=0.825):
        return False

    min_hw = min(
        min(last_rrect_in_line.getWidth(), last_rrect_in_line.getHeight()),
        min(rrect.getWidth(), rrect.getHeight()),
    )
    if abs(last_rrect_in_line.getCenterX() - rrect.getCenterX()) > 0.25 * min_hw:
        return False

    last_rrect_bbox = last_rrect_in_line.getBoundingBox()
    cur_rrect_bbox = rrect.getBoundingBox()
    # add some padding for both bboxes in y axis
    last_rrect_bbox.top *= 0.95
    last_rrect_bbox.bottom *= 1.05
    cur_rrect_bbox.top *= 0.95
    cur_rrect_bbox.bottom *= 1.05

    _iou = iou(last_rrect_bbox, cur_rrect_bbox)

    if _iou < iou_thresh:
        return False

    max_last_y = last_rrect_in_line.maxY()
    min_cur_y = rrect.minY()

    if abs(max_last_y - min_cur_y) > 0.5 * min_hw:
        return False

    if not check_merged_size(text_line, rrect):
        return False

    return True


def check_horizontal_mergable(
    text_line: LanyOcrTextLine,
    rrect: LanyOcrRRect,
    iou_thresh=0.0025,
) -> bool:
    last_rrect_in_line = text_line.sub_rrects[-1]

    # print(last_rrect_in_line.rrect, rrect.rrect)

    if not check_area_merge(last_rrect_in_line, rrect, thresh=0.7):
        return False

    min_hw = min(
        min(last_rrect_in_line.getWidth(), last_rrect_in_line.getHeight()),
        min(rrect.getWidth(), rrect.getHeight()),
    )
    if abs(last_rrect_in_line.getCenterY() - rrect.getCenterY()) > 0.35 * min_hw:
        return False

    last_rrect_bbox = last_rrect_in_line.getBoundingBox()
    cur_rrect_bbox = rrect.getBoundingBox()

    # add some padding for both bboxes in y axis
    last_rrect_bbox.left *= 0.95
    last_rrect_bbox.right *= 1.05
    cur_rrect_bbox.left *= 0.95
    cur_rrect_bbox.right *= 1.05

    _iou = iou(last_rrect_bbox, cur_rrect_bbox)

    if _iou < iou_thresh:
        return False

    max_last_x = last_rrect_in_line.maxX()
    min_cur_x = rrect.minX()

    min_hw = min(text_line.avgHeight(), text_line.avgWidth())

    if abs(max_last_x - min_cur_x) > 0.5 * min_hw:
        return False

    return True


def merge_text_boxes_step(
    rrects: List[LanyOcrRRect], merge_type: LanyOcrMergeType
) -> Tuple[List[LanyOcrTextLine], List[LanyOcrTextLine]]:

    multi_rrects_lines: List[LanyOcrTextLine] = []
    single_rrect_lines: List[LanyOcrTextLine] = []

    sorted_rrects: List[LanyOcrRRect] = []

    if merge_type in [LanyOcrMergeType.UPWARD]:
        sorted_rrects = sorted(rrects, reverse=True, key=lambda item: item.rrect[0][1])
    elif merge_type in [LanyOcrMergeType.DOWNWARD]:
        sorted_rrects = sorted(rrects, reverse=False, key=lambda item: item.rrect[0][1])
    elif merge_type in [LanyOcrMergeType.VERTICAL]:
        sorted_rrects = sorted(rrects, reverse=False, key=lambda item: item.rrect[0][1])
    else:
        sorted_rrects = sorted(rrects, reverse=False, key=lambda item: item.rrect[0][0])

    while sorted_rrects:
        base_rrect = sorted_rrects.pop(0)

        text_line = LanyOcrTextLine(sub_rrects=[base_rrect])

        not_merged_rrects = []
        while sorted_rrects:
            cur_rrect = sorted_rrects.pop(0)
            mergable = False
            if merge_type == LanyOcrMergeType.DOWNWARD:
                mergable = check_downward_mergable(text_line, cur_rrect)
            elif merge_type == LanyOcrMergeType.UPWARD:
                mergable = check_upward_mergable(text_line, cur_rrect)
            elif merge_type == LanyOcrMergeType.VERTICAL:
                mergable = check_vertical_mergable(text_line, cur_rrect)
            else:
                mergable = check_horizontal_mergable(text_line, cur_rrect)

            if mergable:
                text_line.sub_rrects.append(cur_rrect)
            else:
                not_merged_rrects.append(cur_rrect)

        # sorted_rrects.extend(not_merged_rrects)
        sorted_rrects = not_merged_rrects

        if len(text_line.sub_rrects) > 1:
            multi_rrects_lines.append(text_line)
        else:
            single_rrect_lines.append(text_line)

    return (multi_rrects_lines, single_rrect_lines)


def merge_text_boxes(
    polys: np.ndarray, merge_vertical: bool = False
) -> List[LanyOcrTextLine]:
    results = []

    all_rrects: List[LanyOcrRRect] = []

    for idx, poly in enumerate(polys):
        cnts = np.array(poly).reshape((1, -1, 2))
        rrect = cv2.minAreaRect(cnts)

        all_rrects.append(
            LanyOcrRRect(
                rrect=rrect,
                points=np.reshape(poly, [-1, 2]).tolist(),
                direction=LanyOcrMergeType.HORIZONTAL,
            )
        )

    (
        horizontal_multi_rrects_lines,
        horizontal_single_rrect_lines,
    ) = merge_text_boxes_step(all_rrects, LanyOcrMergeType.HORIZONTAL)
    results.extend(horizontal_multi_rrects_lines)

    all_rrects.clear()
    for line in horizontal_single_rrect_lines:
        all_rrects.extend(line.sub_rrects)

    upward_multi_rrects_lines, upward_single_rrect_lines = merge_text_boxes_step(
        all_rrects, LanyOcrMergeType.UPWARD
    )
    results.extend(upward_multi_rrects_lines)

    all_rrects.clear()
    for line in upward_single_rrect_lines:
        all_rrects.extend(line.sub_rrects)

    downward_multi_rrects_lines, downward_single_rrect_lines = merge_text_boxes_step(
        all_rrects, LanyOcrMergeType.DOWNWARD
    )
    results.extend(downward_multi_rrects_lines)

    if merge_vertical:
        all_rrects.clear()
        for line in downward_single_rrect_lines:
            all_rrects.extend(line.sub_rrects)

        (
            vertical_multi_rrects_lines,
            vertical_single_rrect_lines,
        ) = merge_text_boxes_step(all_rrects, LanyOcrMergeType.VERTICAL)

        results.extend(vertical_multi_rrects_lines)
        results.extend(vertical_single_rrect_lines)
    else:
        results.extend(downward_single_rrect_lines)

    return results
