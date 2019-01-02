from PIL import Image
import numpy as np

def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
#   img = Image.fromarray(array, mode='RGB')

  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img
