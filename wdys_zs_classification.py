"""This script includes the main class of the paper: What do you see?"""
import torch
from PIL import Image
import clip
import numpy as np
from torchvision import transforms
from typing import Any, Dict, List, Optional, Union, Tuple

MAX_WORDS = 100
CHAR_THRESHOLD = 250

class WDYS:
  """WDYS zero-shot image classifier."""

  def __init__(self, classes_dict: Dict[str, List[Any]],
               options: Dict[str, Union[str, bool]]):
    """Initialization function.

    Args:
      classes_dict: A dictionary of class labels and optionally generated class
        description
      and their embedding features.
      options: A dictionary of model options.
    """
    self.set_device()
    self.transform = None
    self.class_names = classes_dict['class_name']
    if 'class_name_emb' in classes_dict:
      self.class_name_features = classes_dict['class_name_emb']
    else:
      self.class_name_features = None
    if 'a_photo_of_emb' in classes_dict:
      self.a_photo_of_features = classes_dict['a_photo_of_emb']
    else:
      self.a_photo_of_features = None
    if 'class_description' in classes_dict:
      self.class_descriptions = classes_dict['class_description']
    else:
      self.class_descriptions = None
    if 'class_description_emb' in classes_dict:
      self.class_description_features = classes_dict['class_description_emb']
    else:
      self.class_description_features = None

    self.class_embedding_method = options['class_embedding_method']
    self.recompute_features = options['recompute_features']
    self.use_image = options['use_image']
    self.use_gemini_prediction = options['use_gemini_prediction']
    self.use_image_description = options['use_image_description']

    self.use_class_descriptions = False
    self.use_class_name = False
    self.use_a_photo_of_class = False
    print(f'~~\n\n{self.class_embedding_method}\n\n')
    if 'class_description' in self.class_embedding_method.lower():
      self.use_class_descriptions = True
    if 'class_name' in self.class_embedding_method.lower():
      self.use_class_name = True
    if 'a_photo_of_class_name' in self.class_embedding_method.lower():
      self.use_a_photo_of_class = True
    if (self.use_class_descriptions and not self.recompute_features and
        self.class_description_features is None):
      raise NotImplementedError(
        'Cannot find class description features in the dataset. '
        'Make sure that the dataset contains class descriptions\' '
        'embedding features.')
    elif (self.use_class_descriptions and self.recompute_features and
          self.class_descriptions is None):
      raise NotImplementedError(
        'Cannot find class descriptions in the dataset. Make sure that the '
        'dataset contains class descriptions.')
    elif (self.use_class_descriptions and not self.recompute_features and
          self.class_name_features is None):
      raise NotImplementedError(
        'Cannot find class name features in the dataset. Make sure that the '
        'dataset contains class names\' embedding features.')
    elif (self.use_a_photo_of_class and not self.recompute_features and
          self.a_photo_of_features is None):
      raise NotImplementedError(
        'Cannot find "a photo of {class}" features in the dataset. Make sure '
        'that the dataset contains such embedding features.')

    if self.recompute_features:
      self.set_model(options['model_name'])

    self.build_classifier_model()

  def activate_recomputing_features(self):
    """Recomputes features when classifying."""
    self.recompute_features = True
    self.set_model()

  def build_classifier_model(self):
    """Builds a zero-shot classifier model."""
    self.weights = 0
    if self.recompute_features:
      features = []
      if self.use_a_photo_of_class:
        for class_name in self.class_names:
          text_to_encode = f'A photo of {class_name}'
          features.append(
            self.encode_text(text_to_encode).detach().cpu().numpy())
        self.weights += torch.tensor(
          np.array(features), dtype=torch.float32, device=self.device
        ).squeeze(1)
      if self.use_class_name:
        for class_name in self.class_names:
          text_to_encode = class_name
          features.append(
            self.encode_text(text_to_encode).detach().cpu().numpy())
        self.weights += torch.tensor(
          np.array(features), dtype=torch.float32, device=self.device
        ).squeeze(1)
      if self.use_class_descriptions:
        for class_descriptions in self.class_descriptions:
          description_features = []
          for class_description in class_descriptions:
            description_features.append(
              self.encode_gemini_text(class_description).detach().cpu().numpy())
          features.append(description_features)
        self.weights += torch.mean(
          torch.tensor(
            np.array(features), dtype=torch.float32, device=self.device), dim=1
        ).squeeze(1)
    else:
      if self.use_class_name:
        self.weights += torch.tensor(
          np.array(self.class_name_features),
          dtype=torch.float32, device=self.device).squeeze(1)
      if self.use_class_descriptions:
        self.weights += torch.mean(
          torch.tensor(
            np.array(self.class_description_features),
            dtype=torch.float32, device=self.device), dim=1
        ).squeeze(1)
      if self.use_a_photo_of_class:
        self.weights += torch.tensor(
          np.array(self.a_photo_of_features),
          dtype=torch.float32, device=self.device).squeeze(1)
    self.weights /= self.weights.norm(dim=-1, keepdim=True)
    self.weights = torch.transpose(self.weights, 0, 1)
    print(f'Built WDYS classifer for {self.weights.shape[-1]} classes.')

  def set_device(self):
    """Assigns torch device."""
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def set_transform(self):
    """Assigns image transform pre-processing functions."""
    self.transform = transforms.Compose([
      transforms.Lambda(
        lambda img: img.convert("RGB") if img.mode != "RGB" else img),
      transforms.ToTensor(),
      transforms.Resize((224, 224)),
    ])

  def set_model(self, model_name: Optional[str] = 'clip-l/14'
                ):
    """Loads CLIP model.

    Args:
      model_name: CLIP model name. Options are: 'clip-b/32', 'clip-b/16',
      and 'clip-l/14'
    """
    if model_name == 'clip-b/32':
      self.model, _ = clip.load('ViT-B/32', device=self.device)
    elif model_name == 'clip-b/16':
      self.model, _ = clip.load('ViT-B/16', device=self.device)
    elif model_name == 'clip-l/14':
      self.model, _ = clip.load('ViT-L/14', device=self.device)
    else:
      raise NotImplementedError(
        f"Model '{model_name}' is not supported. "
        "Supported models are: 'clip-b/32', 'clip-b/16', 'clip-l/14'")
    self.model.eval()

  def encode_gemini_text(self, text: str) -> torch.Tensor:
    """Encodes text produced by Gemini Pro.

    Args:
      text: Text to encode.
    Returns:
      Encoded feature as torch tensor.
    """
    with torch.no_grad():
      feature = None
      threshold = CHAR_THRESHOLD
      first = True
      while feature is None:
        try:
          feature = self.encode_text(text)
        except:
          if first:
            words = text.split()
            if len(words) > MAX_WORDS:
              text = ' '.join(words[:MAX_WORDS])
            first = False
          else:
            threshold -= 50
            if len(text) > threshold:
              text = text[:threshold]
    return feature

  def encode_text(self, text: str) -> torch.Tensor:
    """Encodes text using CLIP model.

    Args:
      text: Text to encode.
    Returns:
      Encoded feature as torch tensor.
    """
    text_inputs = torch.cat([clip.tokenize(text)]).to(self.device)
    with torch.no_grad():
      embd = self.model.encode_text(text_inputs)
    embd /= embd.norm(dim=-1, keepdim=True)
    return embd

  def encode_image(self, image: np.array) -> torch.Tensor:
    """Encodes image using CLIP model.

    Args:
      image: A numpy array of input image.
    Returns:
      Encoded feature as torch tensor.
    """
    if self.transform is None:
      self.set_transform()
    with torch.no_grad():
      embd = self.model.encode_image(
        self.transform(Image.fromarray(image)).unsqueeze(0))
    embd /= embd.norm(dim=-1, keepdim=True)
    return embd

  def classify(self, image: Optional[np.array] = None,
               image_feature: Optional[np.array] = None,
               image_description: Optional[str] = None,
               description_feature: Optional[np.array] = None,
               gemini_prediction: Optional[str] = None,
               gemini_prediction_feature: Optional[np.array] = None,
               ) -> Tuple[str, List[str]]:
    """Performs zero-shot image classification on given data.

    Args:
      image: A numpy array of input image.
      image_feature: CLIP feature of image.
      image_description: Gemini's image description.
      description_feature: Gemini's image description's CLIP feature.
      gemini_prediction: Gemini's class prediction.
      gemini_prediction_feature: Gemini's class prediction's CLIP feature.
    Returns:
      Predicted class name and a list of the top-5 predicted class names.
    """
    features = []
    if self.recompute_features:
      if image is not None:
        features.append(self.encode_image(image).detach().cpu().numpy())
      if image_description is not None:
        features.append(self.encode_gemini_text(image_description
                                                ).detach().cpu().numpy())
      if gemini_prediction is not None:
        features.append(self.encode_gemini_text(gemini_prediction
                                                ).detach().cpu().numpy())
    else:
      if image_feature is not None:
        features.append(image_feature)
      if description_feature is not None:
        features.append(description_feature.detach().cpu().numpy())
      if gemini_prediction_feature is not None:
        features.append(gemini_prediction_feature.detach().cpu().numpy())
    if not features:
      raise AssertionError('Insufficient input data.')
    in_features = torch.sum(torch.tensor(
      np.array(features), dtype=torch.float32, device=self.device), dim=0)
    in_features /= in_features.norm(dim=-1, keepdim=True)
    logits = in_features @ self.weights
    predicted_class_name = self.class_names[torch.argmax(logits, dim=1)]
    top5_scores, top5_indices = torch.topk(logits, k=5, dim=1)
    top5_class_names = [self.class_names[index] for index in
                        top5_indices.detach().cpu().numpy().squeeze()]
    # logits = logits.flatten()
    # scores = [logits[index] for index in
    #                     top5_indices.detach().cpu().numpy().squeeze()]
    return predicted_class_name, top5_class_names