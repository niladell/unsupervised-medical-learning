from PIL import Image
import numpy as np

def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling.
  For more documentation on the several modes: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#concept-modes
  """
  if array.shape[2]==3:
    img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  # img = Image.fromarray(array, mode='RGB')
  elif array.shape[2]==1:
    # PIL also provides limited support for a few special modes, including LA (L with alpha)
    # https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    img = Image.fromarray(np.uint8((np.squeeze(array) + 1.0) / 2.0 * 255), mode='L')
  else:
    raise ValueError("Don't know how to handle: {}".format(array.shape))


  return img
