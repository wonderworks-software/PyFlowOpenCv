import cv2
import numpy as np

def get_h_w_c(image: np.ndarray):
    """Returns the height, width, and number of channels."""
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return h, w, c

def as_2d_grayscale(img: np.ndarray) -> np.ndarray:
    """Given a grayscale image, this returns an image with 2 dimensions (image.ndim == 2)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    assert False, f"Invalid image shape {img.shape}"

def convert_to_BGRA(img: np.ndarray, in_c: int) -> np.ndarray:
    assert in_c in (1, 3, 4), f"Number of channels ({in_c}) unexpected"
    if in_c == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif in_c == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    return img.copy()

def normalize_image(img: np.ndarray) -> np.ndarray:
    dtype_max = 1
    try:
        dtype_max = np.iinfo(img.dtype).max
    except:
        logger.debug("img dtype is not int")
    return np.clip(img.astype(np.float32) / dtype_max, 0, 1)

def as_target_channels(img: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Given a number of target channels (either 1, 3, or 4), this convert the given image
    to an image with that many channels. If the given image already has the correct
    number of channels, it will be returned as is.
    Only widening conversions are supported.
    """
    c = get_h_w_c(img)[2]

    if target_channels == 1:
        return as_2d_grayscale(img)
    if c == target_channels:
        return img

    assert c < target_channels

    if target_channels == 3:
        if c == 1:
            img = as_2d_grayscale(img)
            return np.dstack((img, img, img))

    if target_channels == 4:
        return convert_to_BGRA(img, c)

    assert False, "Unable to convert image"

def resize_to_fit_rect(img2resize,rect,interpolation = cv2.INTER_AREA):
    i_h, i_w, i_c = get_h_w_c(img2resize)
    r_w, r_h = rect
    aspect = float(i_w)/float(i_h)
    o_w = i_w
    o_h = i_h
    if i_w > r_w:
        o_w = r_w
        factor = float(o_w)/float(i_w)
        o_h = int(o_h*factor)
    if o_h > r_h:
        o_h = r_h
        factor = float(o_h)/float(i_h)
        o_w = int(o_w*factor)
    dim = (o_w,o_h)
    if dim != (i_h,i_w):
        return cv2.resize(img2resize, dim, interpolation = interpolation)
    else:
        return img2resize   
     
def resize_to_fit(img2resize,ref,interpolation = cv2.INTER_AREA):
    r_h, r_w, r_c = get_h_w_c(ref)
    img2resize = resize_to_fit_rect(img2resize,(r_w,r_h),interpolation=interpolation)
    return img2resize

def expand_image_to_fit_rect(img2expand,rect,center = False, copy = False):
    # Pad base image with transparency if necessary to match size with overlay
    b_h, b_w, b_c = get_h_w_c(img2expand)
    o_w, o_h = rect
    max_h = max(b_h, o_h)
    max_w = max(b_w, o_w)    
    top = bottom = left = right = 0
    if b_h < max_h:
        top = (max_h - b_h) // 2
        bottom = max_h - b_h - top
    if b_w < max_w:
        left = (max_w - b_w) // 2
        right = max_w - b_w - left
    if any((top, bottom, left, right)):
        # copyMakeBorder will create black border if base not converted to RGBA first
        img2expand = convert_to_BGRA(img2expand, b_c)
        if not center:
            bottom = bottom + top 
            right = right + left
            top = left = 0

        img2expand = cv2.copyMakeBorder(
            img2expand, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
        )
    else:  # Make sure cached image not being worked on regardless
        if copy:
            img2expand = img2expand.copy()
        else:
            img2expand = img2expand

    return img2expand

def expand_image_to_fit(img2expand,ref,center = True , copy = True):
    # Pad base image with transparency if necessary to match size with overlay
    o_h, o_w, _ = get_h_w_c(ref)
    img2expand = expand_image_to_fit_rect(img2expand,(o_w, o_h),center,copy)
    return img2expand