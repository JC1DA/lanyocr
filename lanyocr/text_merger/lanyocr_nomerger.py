from typing import List

from lanyocr.lanyocr_utils import LanyOcrRRect
from lanyocr.lanyocr_utils import LanyOcrTextLine
from lanyocr.text_merger import LanyOcrMerger


class LanyOcrNoMerger(LanyOcrMerger):
    def __init__(self) -> None:
        super().__init__()

    def merge_to_lines(
        self,
        rrects: List[LanyOcrRRect],
        merge_rotated: bool = True,
        merge_vertical: bool = True,
    ) -> List[LanyOcrTextLine]:
        lines: List[LanyOcrTextLine] = []

        for rrect in rrects:
            lines.append(
                LanyOcrTextLine(
                    sub_rrects=[rrect],
                    direction="",
                )
            )

        return lines
