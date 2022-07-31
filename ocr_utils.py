import re
from collections import namedtuple

import numpy as np
import Polygon as plg


def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        return raw.decode("utf-8-sig", errors="replace")
    except:
        return None


def validate_point_inside_bounds(x, y, imWidth, imHeight):
    if x < 0 or x > imWidth:
        raise Exception(
            "X value (%s) not valid. Image dimensions: (%s,%s)" % (x, imWidth, imHeight)
        )
    if y < 0 or y > imHeight:
        raise Exception(
            "Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s"
            % (y, imWidth, imHeight)
        )


def validate_clockwise_points(points):
    """
    Validates that the points are in clockwise order.
    """
    edge = []
    for i in range(len(points) // 2):
        edge.append(
            (int(points[(i + 1) * 2 % len(points)]) - int(points[i * 2]))
            * (int(points[((i + 1) * 2 + 1) % len(points)]) + int(points[i * 2 + 1]))
        )
    if sum(edge) > 0:
        raise Exception(
            "Points are not clockwise. The coordinates of bounding points have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards."
        )


def get_tl_line_values(
    line,
    LTRB=True,
    withTranscription=False,
    withConfidence=False,
    imWidth=0,
    imHeight=0,
):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription]
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription]
    Returns values from a textline. Points , [Confidences], [Transcriptions]
    """
    confidence = 0.0
    transcription = ""
    points = []

    numPoints = 4

    if LTRB:

        numPoints = 4

        if withTranscription and withConfidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                line,
            )
            if m == None:
                m = re.match(
                    r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                    line,
                )
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence,transcription"
                )
        elif withConfidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$",
                line,
            )
            if m == None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,confidence"
                )
        elif withTranscription:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,(.*)$",
                line,
            )
            if m == None:
                raise Exception(
                    "Format incorrect. Should be: xmin,ymin,xmax,ymax,transcription"
                )
        else:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,?\s*$",
                line,
            )
            if m == None:
                raise Exception("Format incorrect. Should be: xmin,ymin,xmax,ymax")

        xmin = int(m.group(1))
        ymin = int(m.group(2))
        xmax = int(m.group(3))
        ymax = int(m.group(4))
        if xmax < xmin:
            raise Exception("Xmax value (%s) not valid (Xmax < Xmin)." % (xmax))
        if ymax < ymin:
            raise Exception("Ymax value (%s)  not valid (Ymax < Ymin)." % (ymax))

        points = [float(m.group(i)) for i in range(1, (numPoints + 1))]

        if imWidth > 0 and imHeight > 0:
            validate_point_inside_bounds(xmin, ymin, imWidth, imHeight)
            validate_point_inside_bounds(xmax, ymax, imWidth, imHeight)

    else:

        numPoints = 8

        if withTranscription and withConfidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*,(.*)$",
                line,
            )
            if m == None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence,transcription"
                )
        elif withConfidence:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*([0-1].?[0-9]*)\s*$",
                line,
            )
            if m == None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,confidence"
                )
        elif withTranscription:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,(.*)$",
                line,
            )
            if m == None:
                raise Exception(
                    "Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4,transcription"
                )
        else:
            m = re.match(
                r"^\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*,\s*(-?[0-9]+)\s*$",
                line,
            )
            if m == None:
                raise Exception("Format incorrect. Should be: x1,y1,x2,y2,x3,y3,x4,y4")

        points = [float(m.group(i)) for i in range(1, (numPoints + 1))]

        validate_clockwise_points(points)

        if imWidth > 0 and imHeight > 0:
            validate_point_inside_bounds(points[0], points[1], imWidth, imHeight)
            validate_point_inside_bounds(points[2], points[3], imWidth, imHeight)
            validate_point_inside_bounds(points[4], points[5], imWidth, imHeight)
            validate_point_inside_bounds(points[6], points[7], imWidth, imHeight)

    if withConfidence:
        try:
            confidence = float(m.group(numPoints + 1))
        except ValueError:
            raise Exception("Confidence value must be a float")

    if withTranscription:
        posTranscription = numPoints + (2 if withConfidence else 1)
        transcription = m.group(posTranscription)
        m2 = re.match(r"^\s*\"(.*)\"\s*$", transcription)
        if (
            m2 != None
        ):  # Transcription with double quotes, we extract the value and replace escaped characters
            transcription = m2.group(1).replace("\\\\", "\\").replace('\\"', '"')

    return points, confidence, transcription


def get_tl_line_values_from_file_contents(
    content,
    CRLF=True,
    LTRB=True,
    withTranscription=False,
    withConfidence=False,
    imWidth=0,
    imHeight=0,
    sort_by_confidences=True,
):
    """
    Returns all points, confindences and transcriptions of a file in lists. Valid line formats:
    xmin,ymin,xmax,ymax,[confidence],[transcription]
    x1,y1,x2,y2,x3,y3,x4,y4,[confidence],[transcription]
    """
    pointsList = []
    transcriptionsList = []
    confidencesList = []

    lines = content.split("\r\n" if CRLF else "\n")
    for line in lines:
        line = line.replace("\r", "").replace("\n", "")
        if line != "":
            points, confidence, transcription = get_tl_line_values(
                line, LTRB, withTranscription, withConfidence, imWidth, imHeight
            )
            pointsList.append(points)
            transcriptionsList.append(transcription)
            confidencesList.append(confidence)

    if withConfidence and len(confidencesList) > 0 and sort_by_confidences:
        import numpy as np

        sorted_ind = np.argsort(-np.array(confidencesList))
        confidencesList = [confidencesList[i] for i in sorted_ind]
        pointsList = [pointsList[i] for i in sorted_ind]
        transcriptionsList = [transcriptionsList[i] for i in sorted_ind]

    return pointsList, confidencesList, transcriptionsList


def evaluate_method(
    gt,
    subm,
    evaluationParams={
        "IOU_CONSTRAINT": 0.5,
        "AREA_PRECISION_CONSTRAINT": 0.5,
        "WORD_SPOTTING": False,
        "MIN_LENGTH_CARE_WORD": 3,
        "GT_SAMPLE_NAME_2_ID": "gt_img_([0-9]+).txt",
        "DET_SAMPLE_NAME_2_ID": "res_img_([0-9]+).txt",
        "LTRB": False,
        "CRLF": False,
        "CONFIDENCES": False,
        "SPECIAL_CHARACTERS": "!?.:,*\"()·[]/'",
        "ONLY_REMOVE_FIRST_LAST_CHARACTER": True,
    },
):
    """
    Method evaluate_method: evaluate method and returns the results
        Results. Dictionary with the following values:
        - method (required)  Global method metrics. Ex: { 'Precision':0.8,'Recall':0.9 }
        - samples (optional) Per sample metrics. Ex: {'sample1' : { 'Precision':0.8,'Recall':0.9 } , 'sample2' : { 'Precision':0.8,'Recall':0.9 }
    """

    def polygon_from_points(points, correctOffset=False):
        """
        Returns a Polygon object to use with the Polygon2 class from a list of 8 points: x1,y1,x2,y2,x3,y3,x4,y4
        """

        if (
            correctOffset
        ):  # this will substract 1 from the coordinates that correspond to the xmax and ymax
            points[2] -= 1
            points[4] -= 1
            points[5] -= 1
            points[7] -= 1

        resBoxes = np.empty([1, 8], dtype="int32")
        resBoxes[0, 0] = int(points[0])
        resBoxes[0, 4] = int(points[1])
        resBoxes[0, 1] = int(points[2])
        resBoxes[0, 5] = int(points[3])
        resBoxes[0, 2] = int(points[4])
        resBoxes[0, 6] = int(points[5])
        resBoxes[0, 3] = int(points[6])
        resBoxes[0, 7] = int(points[7])
        pointMat = resBoxes[0].reshape([2, 4]).T
        return plg.Polygon(pointMat)

    def rectangle_to_polygon(rect):
        resBoxes = np.empty([1, 8], dtype="int32")
        resBoxes[0, 0] = int(rect.xmin)
        resBoxes[0, 4] = int(rect.ymax)
        resBoxes[0, 1] = int(rect.xmin)
        resBoxes[0, 5] = int(rect.ymin)
        resBoxes[0, 2] = int(rect.xmax)
        resBoxes[0, 6] = int(rect.ymin)
        resBoxes[0, 3] = int(rect.xmax)
        resBoxes[0, 7] = int(rect.ymax)

        pointMat = resBoxes[0].reshape([2, 4]).T

        return plg.Polygon(pointMat)

    def rectangle_to_points(rect):
        points = [
            int(rect.xmin),
            int(rect.ymax),
            int(rect.xmax),
            int(rect.ymax),
            int(rect.xmax),
            int(rect.ymin),
            int(rect.xmin),
            int(rect.ymin),
        ]
        return points

    def get_union(pD, pG):
        areaA = pD.area()
        areaB = pG.area()
        return areaA + areaB - get_intersection(pD, pG)

    def get_intersection_over_union(pD, pG):
        try:
            return get_intersection(pD, pG) / get_union(pD, pG)
        except:
            return 0

    def get_intersection(pD, pG):
        pInt = pD & pG
        if len(pInt) == 0:
            return 0
        return pInt.area()

    def compute_ap(confList, matchList, numGtCare):
        correct = 0
        AP = 0
        if len(confList) > 0:
            confList = np.array(confList)
            matchList = np.array(matchList)
            sorted_ind = np.argsort(-confList)
            confList = confList[sorted_ind]
            matchList = matchList[sorted_ind]
            for n in range(len(confList)):
                match = matchList[n]
                if match:
                    correct += 1
                    AP += float(correct) / (n + 1)

            if numGtCare > 0:
                AP /= numGtCare

        return AP

    def transcription_match(
        transGt,
        transDet,
        specialCharacters="!?.:,*\"()·[]/'",
        onlyRemoveFirstLastCharacterGT=True,
    ):

        if onlyRemoveFirstLastCharacterGT:
            # special characters in GT are allowed only at initial or final position
            if transGt == transDet:
                return True

            if specialCharacters.find(transGt[0]) > -1:
                if transGt[1:] == transDet:
                    return True

            if specialCharacters.find(transGt[-1]) > -1:
                if transGt[0 : len(transGt) - 1] == transDet:
                    return True

            if (
                specialCharacters.find(transGt[0]) > -1
                and specialCharacters.find(transGt[-1]) > -1
            ):
                if transGt[1 : len(transGt) - 1] == transDet:
                    return True
            return False
        else:
            # Special characters are removed from the begining and the end of both Detection and GroundTruth
            while len(transGt) > 0 and specialCharacters.find(transGt[0]) > -1:
                transGt = transGt[1:]

            while len(transDet) > 0 and specialCharacters.find(transDet[0]) > -1:
                transDet = transDet[1:]

            while len(transGt) > 0 and specialCharacters.find(transGt[-1]) > -1:
                transGt = transGt[0 : len(transGt) - 1]

            while len(transDet) > 0 and specialCharacters.find(transDet[-1]) > -1:
                transDet = transDet[0 : len(transDet) - 1]

            return transGt == transDet

    def include_in_dictionary(transcription):
        """
        Function used in Word Spotting that finds if the Ground Truth transcription meets the rules to enter into the dictionary. If not, the transcription will be cared as don't care
        """
        # special case 's at final
        if (
            transcription[len(transcription) - 2 :] == "'s"
            or transcription[len(transcription) - 2 :] == "'S"
        ):
            transcription = transcription[0 : len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip("-")

        specialCharacters = "'!?.:,*\"()·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, " ")

        transcription = transcription.strip()

        if len(transcription) != len(transcription.replace(" ", "")):
            return False

        if len(transcription) < evaluationParams["MIN_LENGTH_CARE_WORD"]:
            return False

        notAllowed = "×÷·"

        range1 = [ord(u"a"), ord(u"z")]
        range2 = [ord(u"A"), ord(u"Z")]
        range3 = [ord(u"À"), ord(u"ƿ")]
        range4 = [ord(u"Ǆ"), ord(u"ɿ")]
        range5 = [ord(u"Ά"), ord(u"Ͽ")]
        range6 = [ord(u"-"), ord(u"-")]

        for char in transcription:
            charCode = ord(char)
            if notAllowed.find(char) != -1:
                return False

            valid = (
                (charCode >= range1[0] and charCode <= range1[1])
                or (charCode >= range2[0] and charCode <= range2[1])
                or (charCode >= range3[0] and charCode <= range3[1])
                or (charCode >= range4[0] and charCode <= range4[1])
                or (charCode >= range5[0] and charCode <= range5[1])
                or (charCode >= range6[0] and charCode <= range6[1])
            )
            if valid == False:
                return False

        return True

    def include_in_dictionary_transcription(transcription):
        """
        Function applied to the Ground Truth transcriptions used in Word Spotting. It removes special characters or terminations
        """
        # special case 's at final
        if (
            transcription[len(transcription) - 2 :] == "'s"
            or transcription[len(transcription) - 2 :] == "'S"
        ):
            transcription = transcription[0 : len(transcription) - 2]

        # hypens at init or final of the word
        transcription = transcription.strip("-")

        specialCharacters = "'!?.:,*\"()·[]/"
        for character in specialCharacters:
            transcription = transcription.replace(character, " ")

        transcription = transcription.strip()

        return transcription

    perSampleMetrics = {}

    matchedSum = 0

    Rectangle = namedtuple("Rectangle", "xmin ymin xmax ymax")

    numGlobalCareGt = 0
    numGlobalCareDet = 0

    arrGlobalConfidences = []
    arrGlobalMatches = []

    for resFile in gt:
        gtFile = decode_utf8(gt[resFile])
        if gtFile is None:
            raise Exception("The file %s is not UTF-8" % resFile)

        recall = 0
        precision = 0
        hmean = 0
        detCorrect = 0
        iouMat = np.empty([1, 1])
        gtPols = []
        detPols = []
        gtTrans = []
        detTrans = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = (
            []
        )  # Array of Ground Truth Polygons' keys marked as don't Care
        detDontCarePolsNum = (
            []
        )  # Array of Detected Polygons' matched with a don't Care GT
        detMatchedNums = []
        pairs = []

        arrSampleConfidences = []
        arrSampleMatch = []
        sampleAP = 0

        evaluationLog = ""

        (pointsList, _, transcriptionsList,) = get_tl_line_values_from_file_contents(
            gtFile, evaluationParams["CRLF"], evaluationParams["LTRB"], True, False
        )
        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n]
            dontCare = transcription == "###"
            if evaluationParams["LTRB"]:
                gtRect = Rectangle(*points)
                gtPol = rectangle_to_polygon(gtRect)
            else:
                gtPol = polygon_from_points(points)
            gtPols.append(gtPol)
            gtPolPoints.append(points)

            # On word spotting we will filter some transcriptions with special characters
            if evaluationParams["WORD_SPOTTING"]:
                if dontCare == False:
                    if include_in_dictionary(transcription) == False:
                        dontCare = True
                    else:
                        transcription = include_in_dictionary_transcription(
                            transcription
                        )

            gtTrans.append(transcription)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        evaluationLog += (
            "GT polygons: "
            + str(len(gtPols))
            + (
                " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
                if len(gtDontCarePolsNum) > 0
                else "\n"
            )
        )

        if resFile in subm:

            detFile = decode_utf8(subm[resFile])

            (
                pointsList,
                confidencesList,
                transcriptionsList,
            ) = get_tl_line_values_from_file_contents(
                detFile,
                evaluationParams["CRLF"],
                evaluationParams["LTRB"],
                True,
                evaluationParams["CONFIDENCES"],
            )

            for n in range(len(pointsList)):
                points = pointsList[n]
                transcription = transcriptionsList[n]

                if evaluationParams["LTRB"]:
                    detRect = Rectangle(*points)
                    detPol = rectangle_to_polygon(detRect)
                else:
                    detPol = polygon_from_points(points)
                detPols.append(detPol)
                detPolPoints.append(points)
                detTrans.append(transcription)

                if len(gtDontCarePolsNum) > 0:
                    for dontCarePol in gtDontCarePolsNum:
                        dontCarePol = gtPols[dontCarePol]
                        intersected_area = get_intersection(dontCarePol, detPol)
                        pdDimensions = detPol.area()
                        precision = (
                            0 if pdDimensions == 0 else intersected_area / pdDimensions
                        )
                        if precision > evaluationParams["AREA_PRECISION_CONSTRAINT"]:
                            detDontCarePolsNum.append(len(detPols) - 1)
                            break

            evaluationLog += (
                "DET polygons: "
                + str(len(detPols))
                + (
                    " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
                    if len(detDontCarePolsNum) > 0
                    else "\n"
                )
            )

            if len(gtPols) > 0 and len(detPols) > 0:
                # Calculate IoU and precision matrixs
                outputShape = [len(gtPols), len(detPols)]
                iouMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtPols), np.int8)
                detRectMat = np.zeros(len(detPols), np.int8)
                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        pG = gtPols[gtNum]
                        pD = detPols[detNum]
                        iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

                for gtNum in range(len(gtPols)):
                    for detNum in range(len(detPols)):
                        if (
                            gtRectMat[gtNum] == 0
                            and detRectMat[detNum] == 0
                            and gtNum not in gtDontCarePolsNum
                            and detNum not in detDontCarePolsNum
                        ):
                            if (
                                iouMat[gtNum, detNum]
                                > evaluationParams["IOU_CONSTRAINT"]
                            ):
                                gtRectMat[gtNum] = 1
                                detRectMat[detNum] = 1
                                # detection matched only if transcription is equal
                                if evaluationParams["WORD_SPOTTING"]:
                                    correct = (
                                        gtTrans[gtNum].upper()
                                        == detTrans[detNum].upper()
                                    )
                                else:
                                    correct = (
                                        transcription_match(
                                            gtTrans[gtNum].upper(),
                                            detTrans[detNum].upper(),
                                            evaluationParams["SPECIAL_CHARACTERS"],
                                            evaluationParams[
                                                "ONLY_REMOVE_FIRST_LAST_CHARACTER"
                                            ],
                                        )
                                        == True
                                    )
                                detCorrect += 1 if correct else 0
                                if correct:
                                    detMatchedNums.append(detNum)
                                pairs.append(
                                    {"gt": gtNum, "det": detNum, "correct": correct}
                                )
                                evaluationLog += (
                                    "Match GT #"
                                    + str(gtNum)
                                    + " with Det #"
                                    + str(detNum)
                                    + " trans. correct: "
                                    + str(correct)
                                    + "\n"
                                )

            if evaluationParams["CONFIDENCES"]:
                for detNum in range(len(detPols)):
                    if detNum not in detDontCarePolsNum:
                        # we exclude the don't care detections
                        match = detNum in detMatchedNums

                        arrSampleConfidences.append(confidencesList[detNum])
                        arrSampleMatch.append(match)

                        arrGlobalConfidences.append(confidencesList[detNum])
                        arrGlobalMatches.append(match)

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        if numGtCare == 0:
            recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
            sampleAP = precision
        else:
            recall = float(detCorrect) / numGtCare
            precision = 0 if numDetCare == 0 else float(detCorrect) / numDetCare
            if evaluationParams["CONFIDENCES"]:
                sampleAP = compute_ap(arrSampleConfidences, arrSampleMatch, numGtCare)

        hmean = (
            0
            if (precision + recall) == 0
            else 2.0 * precision * recall / (precision + recall)
        )

        matchedSum += detCorrect
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics[resFile] = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean,
            "pairs": pairs,
            "AP": sampleAP,
            "iouMat": [] if len(detPols) > 100 else iouMat.tolist(),
            "gtPolPoints": gtPolPoints,
            "detPolPoints": detPolPoints,
            "gtTrans": gtTrans,
            "detTrans": detTrans,
            "gtDontCare": gtDontCarePolsNum,
            "detDontCare": detDontCarePolsNum,
            "evaluationParams": evaluationParams,
            "evaluationLog": evaluationLog,
        }

    # Compute AP
    AP = 0
    if evaluationParams["CONFIDENCES"]:
        AP = compute_ap(arrGlobalConfidences, arrGlobalMatches, numGlobalCareGt)

    methodRecall = 0 if numGlobalCareGt == 0 else float(matchedSum) / numGlobalCareGt
    methodPrecision = (
        0 if numGlobalCareDet == 0 else float(matchedSum) / numGlobalCareDet
    )
    methodHmean = (
        0
        if methodRecall + methodPrecision == 0
        else 2 * methodRecall * methodPrecision / (methodRecall + methodPrecision)
    )

    methodMetrics = {
        "precision": methodPrecision,
        "recall": methodRecall,
        "hmean": methodHmean,
        "AP": AP,
    }

    resDict = {
        "calculated": True,
        "Message": "",
        "method": methodMetrics,
        "per_sample": perSampleMetrics,
    }

    return resDict
