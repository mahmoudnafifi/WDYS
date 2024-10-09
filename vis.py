"""Visualization code used in the paper: What do you see?"""

import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import argparse
import os
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pickle
from prettytable import PrettyTable
import wdys_zs_classification as wdys

EPS = np.finfo(float).eps
def mask_text(text: str, kernel_size: Optional[int] = 3,
              stride: Optional[int] = 1
              ) -> Tuple[List[str], List[Tuple[int, int]]]:
  """Generates masked text using a sliding kernel.

  Args:
    text: Input text.
    kernel_size: Size of the sliding kernel (words).
    stride: Stride for sliding the kernel (words).

  Returns:
    A list of masked texts and indices of masked out words in the original text.
  """
  words = text.split()
  masked_texts = []
  masked_indices = []

  for i in range(0, len(words) - kernel_size + 1, stride):
    masked_words = [' ' * len(word) if i <= j < i + kernel_size else word for
                    j, word in enumerate(words)]
    masked_text = ' '.join(masked_words)
    masked_texts.append(masked_text)
    masked_indices.append((i, i + kernel_size))
  return masked_texts, masked_indices

def create_heatmap_image(
    text: str, heatmap_values: np.array,
    output_path: str, image_width: Optional[int] = 800,
    image_height: Optional[int] = 200,
    font_size: Optional[int] = 24):
  """Generates an image with text colored according to its significance.

  Args:
    text: Input text.
    heatmap_values: Heatmap as np array.
    output_path: Path to save colored text image.
    image_width: Width of output image.
    image_height: Height of output image.
    font_size: Size of text font.
  """

  img = Image.new('RGB', (image_width, image_height), color='white')
  draw = ImageDraw.Draw(img)
  font = ImageFont.truetype("arial.ttf", font_size)
  words = text.split()
  word_widths = [draw.textlength(word, font=font) for word in words]
  total_text_width = sum(word_widths)
  x_start = (image_width - total_text_width) // 2
  y_start = (image_height - font_size) // 2
  x_pos = x_start
  for word, heatmap_value, word_width in zip(words, heatmap_values, word_widths
                                             ):
    color = (round(heatmap_value * 255), 0, 0)
    draw.text((x_pos, y_start), word, fill=color, font=font)
    x_pos += word_width + 5
  img.save(output_path)


def mask_tile(image: np.ndarray, tile_index: int,
              kernel_size: int, stride: int) -> Tuple[np.array, np.array]:
  """Mask a specific tile in the image with zeros.

  Args:
    image: Input image array.
    tile_index: Index of the tile to mask.
    kernel_size: Size of the square kernel.
    stride: Stride for sliding the kernel.

  Returns:
    Image array with the specified tile masked out and corresponding heat map.
  """
  h, w, _ = image.shape
  padded_image = np.pad(image, ((0, kernel_size),
                                (0, kernel_size), (0, 0)),
                        mode='reflect')
  tile_row = tile_index // ((h - kernel_size) // stride + 1)
  tile_col = tile_index % ((w - kernel_size) // stride + 1)
  start_row = tile_row * stride
  end_row = start_row + kernel_size
  start_col = tile_col * stride
  end_col = start_col + kernel_size
  mask = np.ones_like(padded_image)
  mask[start_row:end_row, start_col:end_col, :] = 0
  masked_image = padded_image * mask
  heat_map = np.zeros(masked_image.shape[:2])
  heat_map[start_row:end_row, start_col:end_col] = 1
  masked_image = masked_image[:h, :w, :]
  heat_map = heat_map[:h, :w]
  return masked_image, heat_map



def test_masked_images(image: np.array, model: object,
                       image_description: str,
                       init_prediction: str,
                       kernel_size: Optional[int] = 50,
                       stride: Optional[int] = 10) -> Tuple[np.array, str]:
  """Test versions of the input image with each tile masked out.

  Args:
    image: Input image array.
    model: WDYS model.
    image_description: Gemini's image description.
    init_prediction: Gemini's initial prediction.
    kernel_size: Size of the square kernel.
    stride: Stride for sliding the kernel.

  Returns:
    heat map highlight most parts that contributed in prediction and initial
    prediction.
  """
  h, w, C = image.shape
  total_tiles = ((h - kernel_size) // stride + 1) * (
      (w - kernel_size) // stride + 1)

  initial_predicted_class, _ = model.classify(
    image=image, image_description=image_description,
    gemini_prediction=init_prediction)
  print(f'Predicted class = {initial_predicted_class}')
  heat_map = np.zeros(image.shape[:2])
  for tile_index in range(total_tiles):
    masked_image, heatmap_i = mask_tile(image, tile_index, kernel_size, stride)
    predicted_class, _ = model.classify(image=masked_image,
                                        image_description=image_description,
                                        gemini_prediction=init_prediction)
    if predicted_class != initial_predicted_class:
      heat_map += heatmap_i
  return heat_map, initial_predicted_class

def get_args():
  parser = argparse.ArgumentParser(description='Visualization.')
  parser.add_argument('--dataset_name', type=str,
                      help='Dataset name.')
  parser.add_argument('--data_folder', type=str,
                      help='Folder of datasets.')
  parser.add_argument('--out_folder', type=str,
                      help='Output folder.')
  parser.add_argument('--file_index', type=int,
                      help='Image file index.')
  parser.add_argument('--image_index', type=int,
                      help='Index of image to visualize.')
  parser.add_argument(
    '--model_name', type=str, default='clip-l/14',
    help='Cross-modal embedding encoder model name.',
    choices=['clip-l/14', 'clip-b/16', 'clip-b/32']
  )
  parser.add_argument(
    '--use_class_descriptions', action='store_true',
    help='To use features of class descriptions produced by Gemini Pro.')
  parser.add_argument('--use_class_name', action='store_true',
                      help='To use features of class names.')
  parser.add_argument('--use_a_photo_of', action='store_true',
                      help='To use features of "A photo of {class name}.')
  return parser.parse_args()

if __name__ == '__main__':
  args = get_args()
  data_folder = args.data_folder
  out_folder = args.out_folder
  dataset_name = args.dataset_name
  model_name = args.model_name
  use_image_features = True
  use_description_features = True
  use_predicted_class_features = True
  image_index = args.image_index
  file_index = args.file_index

  if not os.path.exists(os.path.join(data_folder, dataset_name, 'classes.pkl')):
    raise FileNotFoundError(
      'The dataset class labels file could not be found: '
      f"{os.path.join(data_folder, dataset_name, 'classes.pkl')}")

  if not os.path.exists(os.path.join(data_folder, dataset_name, 'images')):
    raise FileNotFoundError('The image folder could not be found: '
                            f"{os.path.join(data_folder, dataset_name)}")

  if sum([args.use_class_descriptions, args.use_class_name, args.use_a_photo_of]
         ) != 1:
    raise AssertionError(
      "One of the options '--use_class_descriptions', '--use_class_name', "
      "'--use_a_photo_of' must be used.")

  if args.use_class_name:
    class_embedding_method = 'class_name'
  elif args.use_a_photo_of:
    class_embedding_method = 'a_photo_of_class_name'
  elif args.use_class_descriptions:
    class_embedding_method = 'class_description'
  else:
    class_embedding_method = None

  experiment_info = {
    'data_folder': data_folder,
    'dataset': dataset_name,
    'class_embedding_method': class_embedding_method,
    'model_name': model_name,
    'image_index': image_index,
    'file_index': file_index,
    'use_image': use_image_features,
    'use_image_description': use_description_features,
    'use_gemini_prediction': use_predicted_class_features,
    'recompute_features': False,
  }

  table = PrettyTable()
  table.field_names = ['Key', 'Value']
  for key, value in experiment_info.items():
    table.add_row([key, value])
  print(table)

  os.makedirs(out_folder, exist_ok=True)
  print(f'Loading class labels data ...')
  with open(os.path.join(data_folder, dataset_name, 'classes.pkl'), 'rb') as f:
    dataset_classes = pickle.load(f)
  print(f'Done!')

  print('Building WDYS zero-shot classification model.')
  model = wdys.WDYS(dataset_classes, experiment_info)
  print(f'Done!')

  testing_file = os.path.join(data_folder, dataset_name, 'images',
                              f'dataframe_{file_index:06d}.pkl')

  print('Processing...')
  with open(testing_file, 'rb') as f:
    image_data = pickle.load(f)
    length = len(image_data['image_bytes'])
    if image_index >= length:
      raise EOFError(f'Image index ({image_index}) is out of range.')
    gt = image_data['gt_classes'][image_index].replace('\n', '')
    image_bytes = image_data['image_bytes'][image_index]
    image = np.array(Image.open(io.BytesIO(image_bytes)))

    image_description = image_data['image_descriptions'][image_index]
    init_prediction = image_data['gemini_predictions'][image_index]

    print(f'image_description = {image_description}')
    print(f'init_prediction = {init_prediction}')
    print(f'ground-truth = {gt}')

    model.activate_recomputing_features()

    clip_predicted_class, _ = model.classify(image=image)
    print(f'clip prediction = {clip_predicted_class}')

    kernel_size = 50
    stride = 10
    heatmap = 0
    while np.sum(heatmap) == 0 and kernel_size <= 200:
      heatmap, prediction = test_masked_images(
        image, model, image_description, init_prediction,
        kernel_size=kernel_size, stride=stride)
      heatmap = heatmap / np.fmax(np.max(heatmap), EPS)
      kernel_size += 50


    heatmap_rgb = plt.cm.hot(heatmap)[..., :3] * 255
    final_image = heatmap_rgb * 0.5 + image * 0.5
    image_pil = Image.fromarray(final_image.astype(np.uint8))
    image_pil.save(os.path.join(
      out_folder,
      f'image_hm_{dataset_name}_{file_index}_{image_index}.png'))
    Image.fromarray(image.astype(np.uint8)).save(
      os.path.join(
        out_folder,
        f'image_{dataset_name}_{file_index}_{image_index}.png'))

    kernel_size = 3
    image_description_heatmap = np.zeros(len(image_description.split()))
    while np.sum(image_description_heatmap) == 0 and kernel_size >= 1:
      masked_texts, masked_indices = mask_text(image_description,
                                               kernel_size=kernel_size)
      for idx, (masked_text, indices) in enumerate(
          zip(masked_texts, masked_indices)):
        predicted_class, _ = model.classify(
          image=image, image_description=masked_text,
          gemini_prediction=init_prediction)

        if predicted_class != prediction:
          start, end = indices
          image_description_heatmap[start:end] += 1

      image_description_heatmap = image_description_heatmap / np.fmax(
        np.max(image_description_heatmap), EPS)
      kernel_size -= 1

    create_heatmap_image(
      image_description, image_description_heatmap,
      os.path.join(
        out_folder,
        f'image_description_{dataset_name}_{file_index}_{image_index}.png'),
      image_width=15 * len(image_description),
    )

    masked_texts, masked_indices = mask_text(init_prediction,
                                             kernel_size=1, stride=1)
    init_prediction_heatmap = np.zeros(len(init_prediction.split()))
    for idx, (masked_text, indices) in enumerate(
        zip(masked_texts, masked_indices
            )):
      predicted_class, _ = model.classify(
        image=image, image_description=image_description,
        gemini_prediction=masked_text)

      if predicted_class != prediction:
        start, end = indices
        init_prediction_heatmap[start:end] += 1
    init_prediction_heatmap = init_prediction_heatmap / np.fmax(np.max(
      init_prediction_heatmap), EPS)
    create_heatmap_image(
      init_prediction, init_prediction_heatmap,
      os.path.join(
        out_folder,
        f'init_prediction_{dataset_name}_{file_index}_{image_index}.png'))


