""" Set of functions to adapt the data, as croping, resizing, formatting, etc.

Some functions related to downloading and adapting the celebA dataset have been modified from 'nmhkahn/DCGAN-tensorflow-slim' (Github repo)
"""


def center_crop( im,
                output_height,
                output_width ):
    h, w = im.shape[:2]
    if h < output_height and w < output_width:
        raise ValueError("The image is too small ({}, {}) for that size ({}, {})"\
                    .format(h, w, output_height, output_width))

    offset_h = int((h - output_height) / 2)
    offset_w = int((w - output_width) / 2)
    return im[offset_h:offset_h+output_height,
                offset_w:offset_w+output_width, :]

