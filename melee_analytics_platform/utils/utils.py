import re

def text_from_prediction(img, box, ocr_reader, numbers_only = False):
    results = ocr_reader.readtext(img[box[1]:box[3], box[0]:box[2]])
    
    if len(results)> 0:
        results = results[0][1]

        if numbers_only:
            results = re.sub("[^0-9]", "", str(results))

        return results
    else:
        return None
    
def get_midpoint(box):
    """
    Given a box prediction output in the format [x1, y1, x2, y2] - return the midpoint
    of the box
    """
    return [box[0] + int((box[2] - box[0])/2), box[1] + int((box[3] - box[1])/2)]