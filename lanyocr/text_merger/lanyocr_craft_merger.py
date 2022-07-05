from typing import List, Tuple
from lanyocr.text_merger import LanyOcrMerger
from lanyocr.lanyocr_utils import (
    LanyOcrMergeType,
    LanyOcrRRect,
    LanyOcrTextLine,
    check_area_merge,
    check_merged_size,
    iou,
)


class LanyOcrCraftBasedMerger(LanyOcrMerger):
    def __init__(self) -> None:
        super().__init__()

    def merge_to_lines(
        self,
        rrects: List[LanyOcrRRect],
        merge_rotated: bool = True,
        merge_vertical: bool = True,
    ) -> List[LanyOcrTextLine]:
        return self.merge_text_boxes(rrects, merge_rotated, merge_vertical)

    # Internal functions

    def check_downward_mergable(
        self,
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

        if (
            iou(rrect.getBoundingBox(), last_rrect_in_line.getBoundingBox())
            < iou_thresh
        ):
            return False

        max_last_x = last_rrect_in_line.maxX()
        min_cur_x = rrect.minX()

        if min_cur_x - max_last_x > min_hw * width_thresh:
            return False

        return True

    def check_upward_mergable(
        self,
        text_line: LanyOcrTextLine,
        rrect: LanyOcrRRect,
        height_thresh=0.75,
        width_thresh=0.5,
        angle_thresh=20,
        iou_thresh=0.005,
    ) -> bool:
        last_rrect_in_line = text_line.sub_rrects[-1]

        if not check_area_merge(last_rrect_in_line, rrect, thresh=0.75):
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
        self,
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
        self,
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
        self, rrects: List[LanyOcrRRect], merge_type: LanyOcrMergeType
    ) -> Tuple[List[LanyOcrTextLine], List[LanyOcrTextLine]]:

        multi_rrects_lines: List[LanyOcrTextLine] = []
        single_rrect_lines: List[LanyOcrTextLine] = []

        sorted_rrects: List[LanyOcrRRect] = []

        if merge_type in [LanyOcrMergeType.UPWARD]:
            sorted_rrects = sorted(
                rrects, reverse=True, key=lambda item: item.rrect[0][1]
            )
        elif merge_type in [LanyOcrMergeType.DOWNWARD]:
            sorted_rrects = sorted(
                rrects, reverse=False, key=lambda item: item.rrect[0][1]
            )
        elif merge_type in [LanyOcrMergeType.VERTICAL]:
            sorted_rrects = sorted(
                rrects, reverse=False, key=lambda item: item.rrect[0][1]
            )
        else:
            sorted_rrects = sorted(
                rrects, reverse=False, key=lambda item: item.rrect[0][0]
            )

        while sorted_rrects:
            base_rrect = sorted_rrects.pop(0)

            text_line = LanyOcrTextLine(sub_rrects=[base_rrect])
            text_line.direction = merge_type

            not_merged_rrects = []
            while sorted_rrects:
                cur_rrect = sorted_rrects.pop(0)
                mergable = False
                if merge_type == LanyOcrMergeType.DOWNWARD:
                    mergable = self.check_downward_mergable(text_line, cur_rrect)
                elif merge_type == LanyOcrMergeType.UPWARD:
                    mergable = self.check_upward_mergable(text_line, cur_rrect)
                elif merge_type == LanyOcrMergeType.VERTICAL:
                    mergable = self.check_vertical_mergable(text_line, cur_rrect)
                else:
                    mergable = self.check_horizontal_mergable(text_line, cur_rrect)

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
        self,
        rrects: List[LanyOcrRRect],
        merge_rotated: bool = True,
        merge_vertical: bool = False,
    ) -> List[LanyOcrTextLine]:
        results = []

        all_rrects = [rrect for rrect in rrects]

        (
            horizontal_multi_rrects_lines,
            horizontal_single_rrect_lines,
        ) = self.merge_text_boxes_step(all_rrects, LanyOcrMergeType.HORIZONTAL)
        results.extend(horizontal_multi_rrects_lines)

        if not merge_rotated and not merge_vertical:
            results.extend(horizontal_single_rrect_lines)
            return results

        all_rrects.clear()
        for line in horizontal_single_rrect_lines:
            all_rrects.extend(line.sub_rrects)

        if merge_rotated:
            (
                upward_multi_rrects_lines,
                upward_single_rrect_lines,
            ) = self.merge_text_boxes_step(all_rrects, LanyOcrMergeType.UPWARD)
            results.extend(upward_multi_rrects_lines)

            all_rrects.clear()
            for line in upward_single_rrect_lines:
                all_rrects.extend(line.sub_rrects)

            (
                downward_multi_rrects_lines,
                downward_single_rrect_lines,
            ) = self.merge_text_boxes_step(all_rrects, LanyOcrMergeType.DOWNWARD)
            results.extend(downward_multi_rrects_lines)

            if not merge_vertical:
                results.extend(downward_single_rrect_lines)
                return results

            all_rrects.clear()
            for line in downward_single_rrect_lines:
                all_rrects.extend(line.sub_rrects)

        (
            vertical_multi_rrects_lines,
            vertical_single_rrect_lines,
        ) = self.merge_text_boxes_step(all_rrects, LanyOcrMergeType.VERTICAL)

        results.extend(vertical_multi_rrects_lines)
        results.extend(vertical_single_rrect_lines)

        return results
