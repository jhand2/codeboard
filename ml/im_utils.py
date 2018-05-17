import cv2
import numpy as np

def parse_code_im(im, pixel_side):
    """
    Parses an image of a code snippet into discrete character images
    """
    # TODO: Cut out code chunk, possibly using detectRegions
    image_size = im.shape[0] * im.shape[1]
    im_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(im_bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 9)

    lines = _segment_lines(thresh)
    line_im_cpy = im.copy()

    words = []
    for (x,y,w,h) in lines:
        # Cutout line from original picture
        l = thresh[y:y+h, x:x+w]

        # TODO: We actually want to separate out words so we know which line
        #       they belong to
        words += _segment_words(l, x, y)

    characters = []
    for (x, y, w, h) in words:
        w = thresh[y:y+h, x:x+w]
        characters += _segment_chars(w, x, y)

    for (x, y, w, h) in words:
        cv2.rectangle(line_im_cpy, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=1)

    cv2.imshow("Words", line_im_cpy)
    k = cv2.waitKey(0)

    cv2.destroyAllWindows()

def _segment_block(bw_img):
    """
    TODO: Could be used to reduce false positives or process images w/ multiple
          code blocks. Not ready for prime time
    """
    kernel = np.ones((50,50), np.uint8)
    im_dilation = cv2.dilate(bw_img, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(im_dilation.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    im_area = bw_img.shape[0] * bw_img.shape[1]
    
    return [cv2.boundingRect(c) for c in contours]


def _segment_lines(bw_img):
    """
    TODO: Remove more false positives

    Private function to parse a full bw code image and return bounding box
    coordinates for each line of text in the image.

    Note: This may return boxes which contain no text but will be easily
    distinguisable as non-characters

    params:
        bw_img: A full black and white image of whiteboard code

    returns:
        a list of bounding boxe tuples in the form [(x, y, w, h)]
    """

    # Dilate image vertically to make lines run together
    kernel = np.ones((1,50), np.uint8)
    im_dilation = cv2.dilate(bw_img, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(im_dilation.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    im_area = bw_img.shape[0] * bw_img.shape[1]
    
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = np.array([(b[2] * b[3]) for b in boxes])

    # First remove obvious outliers. Either bigger than 10% of the image
    # or smaller than 0.1% of the image.
    # We can remove the other false positives later if we can't parse
    # any characters from them
    keep = (areas >= (im_area * 0.001)) & (areas <= (im_area * 0.1))
    areas = areas[keep]
    boxes = [boxes[i] for i, v in enumerate(keep) if v]

    return boxes
    

def _segment_words(line_im, x_line, y_line):
    """
    TODO: We sould probably not take x_line and y_line. Make the caller
          do that work

    Private function to parse a bw line of text and return bounding box
    coordinates for each word of text in the image.

    Note: This may return boxes which contain no text but will be easily
    distinguisable as non-characters

    params:
        line_im : A black and white image of a line of text
        x_line  : the leftmost x coordinate of line_im in the larger image
        y_line  : the topmost y coordinate of line_im in the larger image

    returns:
        a list of bounding boxe tuples in the form [(x, y, w, h)]
    """

    # TODO: Not sure if this can be static. May need to chose different
    #       numbers based on image size
    kernel = np.ones((15,10), np.uint8)
    im_dilation = cv2.dilate(line_im, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(im_dilation.copy(),
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes.sort(key=lambda b : -(b[2] * b[3]))

    # Decide which boxes actually bound characters
    mat = np.zeros(line_im.shape, dtype=np.uint8)
    keep = np.ones((len(boxes), 1))

    # Mark box locations in overall matrix
    for i in range(len(boxes)):
        (x, y, w, h) = boxes[i]
        mat[y:y+h, x:x+w] = i + 1

    # Check for contained boxes
    for i in range(len(boxes)):
        (x, y, w, h) = boxes[i]
        a = w * h
        overlapping = np.unique(mat[y:y+h,x:x+w])
        for idx in overlapping:
            if idx != (i + 1):
                (x2, y2, w2, h2) = boxes[idx - 1]
                a2 = w2 * h2

                # Calculate percent of smaller that overlaps bigger
                # Note: a is always bigger due to sorting
                overlap = np.sum(mat[y:y+h,x:x+w] == idx) / a2

                # Some boxes may have very slight overlap. We don't want
                # to count those. Check if over 80% overlap
                if overlap > 0.8:
                    keep[idx - 1] = 0

    # Convert to coordinate system of the original image
    translated = [(b[0]+x_line, b[1]+y_line, b[2], b[3]) for i, b in enumerate(boxes) if keep[i]]

    return translated

def _segment_chars(w_im, x, y):
    """
    TODO: Implement character sgmentation algorithm in this paper
          https://www.mobt3ath.com/uplode/book/book-2963.pdf
    """

    return []

