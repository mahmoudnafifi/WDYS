# WDYS
PyTorch implementation of the paper: 

[What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Vision-Language Models](https://arxiv.org/abs/2405.15668). 

[Abdelrahman Abdelhamed](https://abdokamel.github.io/)<sup>*1</sup>, [Mahmoud Afifi](https://www.mafifi.info/)<sup>*2</sup>, [Alec Go](https://scholar.google.com/citations?user=I54RNh4AAAAJ&hl=en&oi=sra)<sup>1</sup>

<sup>*</sup>*Equal contribution*

<sup>1</sup>Google Research  &ensp; <sup>2</sup>Google




<p align="center">
  <img src="https://github.com/user-attachments/assets/d9743225-29f6-40ef-b1f7-f3f1be5af401" width="40%">
</p>



## Abstract

Large language models (LLMs) has been effectively used for many computer vision tasks, including image classification. In this paper, we present a simple yet effective approach for zero-shot image classification using multimodal LLMs. By employing multimodal LLMs, we generate comprehensive textual representations from input images. These textual representations are then utilized to generate fixed-dimensional features in a cross-modal embedding space. Subsequently, these features are fused together to perform zero-shot classification using a linear classifier. Our method does not require prompt engineering for each dataset; instead, we use a single, straightforward, set of prompts across all datasets. We evaluated our method on several datasets, and our results demonstrate its remarkable effectiveness, surpassing benchmark accuracy on multiple datasets. On average over ten benchmarks, our method achieved an accuracy gain of 4.1 percentage points, with an increase of 6.8 percentage points on the ImageNet dataset, compared to prior methods. Our findings highlight the potential of multimodal LLMs to enhance computer vision tasks such as zero-shot image classification, offering a significant improvement over traditional methods.

## Method

![teaser](https://github.com/user-attachments/assets/cece737b-c847-4c1a-aa18-cfd367fb9415)

We propose a zero-shot image classification method that leverages multimodal large language models (LLMs) to enhance the accuracy of standard zero-shot classification. Our method
employs a set of engineered prompts to generate image description and initial class prediction by
the LLM. Subsequently, we encode this data along with the input testing image using a cross-modal
embedding encoder to project the inputs into a common feature space. Finally, we fuse the generated features to produce the final query feature, which is then utilized by a standard zero-shot linear
image classifier to predict the final class.

## Datasets
We provide a `.zip` file for each dataset used in the paper. Each dataset contains the following:

### 1. `classes.pkl`
The `classes.pkl` file includes:
1. `class_name`: A list of class names.
2. `class_name_emb`: A list of class names represented in the embedding space.
3. `a_photo_of_emb`: A list of class names in the `'a_photo_of_{class}'` template, represented in the embedding space.
4. `class_description`: A list of class descriptions generated by the LLM. Each element is an array of 50 descriptions corresponding to the class name.
5. `class_description_emb`: The embedding representation of the class descriptions.

### 2. `images` folder
The `images` folder contains the dataset's test images, stored in `.pkl` files. Each `.pkl` file includes:
1. `image_bytes`: A list of test image bytes, where each element represents a single test image.
2. `image_features`: A list of embedding representations of the test images.
3. `image_description`: The LLM's output in response to the query "What do you see?", as described in the paper.
4. `image_description_features`: A list of tensors, each representing the embedding of an image description.
5. `gemini_predictions`: Initial predictions generated by the LLM.
6. `gemini_prediction_features`: Embedding representations of the initial predictions.
7. `gt_classes`: Ground-truth class names.
8. `gemini_eval`: The evaluation of the initial predictions, where the LLM was asked whether its predicted class matches the ground-truth.

<div align="center">
  
| Dataset Name | Download Link |
|--------------|---------------|
| Sun397       | [Download](https://drive.google.com/file/d/1oCB-OdGYOepLsT1fYm5AwgvLSDHSfhyQ/view?usp=sharing)  |
| Places       | [Download](https://drive.google.com/file/d/1wAn4n4Lf91dPr9O2wsKJ3S655k9N497u/view?usp=sharing)  |
| Pets         | [Download](https://drive.google.com/file/d/1lJSIsuWPkq53NvNJO_WLQLqLfOQ-1UrP/view?usp=sharing)  |
| ImageNet     | [Download](https://drive.google.com/file/d/1VR279ucfgR0Qf7wD7jHGQzRRoJbvoZrV/view?usp=sharing)  |
| Food         | [Download](https://drive.google.com/file/d/13wZelrlbAec9mdMPHtwFAa_yAkIjUnHk/view?usp=sharing)  |
| DTD          | [Download](https://drive.google.com/file/d/1Y0BZDDZP1DJ3KibDI_6Ua6m9wMq2FDJm/view?usp=sharing)  |
| CIFAR-100    | [Download](https://drive.google.com/file/d/14N5w73k-vrWbbYja3CI2nesKBBJxbnSf/view?usp=sharing)  |
| CIFAR-10     | [Download](https://drive.google.com/file/d/1jNWNnNDkOE4mW_Qei6IWNVqkg7hivFnh/view?usp=sharing)  |
| Cars         | [Download](https://drive.google.com/file/d/13es8l17J4eKc0UotgU_AP-8GH-3R2kMm/view?usp=sharing)  |
| Caltech      | [Download](https://drive.google.com/file/d/1cRIem01NL72Oh5VGURH2_dM8uNVYGwBS/view?usp=sharing)  |

</div>

## Code
Example use: 
```python
python main --dataset_name cifar10 --data_folder path/to/dataset/folder --use_image_features --use_description_features --use_predicted_class_features --use_class_descriptions
```

In this example, we test on the CIFAR-10 testing set, where the dataset folder should be located in `path/to/dataset/folder`. The flags `--use_image_features`, `--use_description_features`, and `--use_predicted_class_features` indicate that we are using the three features described in the paper. The flag `--use_class_descriptions` specifies that class descriptions generated by the LLM will be used to match the fused feature with the features of class descriptions. Alternatively, you can use `--use_class_name` to use class names instead, or `--use_a_photo_of` to apply the template `'A photo of {class}'`. 

**Note that this implementation is unofficial and provided for research and experimental purposes.**


## Citation
If you use this code, please cite our paper:
```
@article{abdelhamed2024WDYS,
  title={What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Large Language Models},
  author={Abdelhamed, Abdelrahman and Afifi, Mahmoud and Go, Alec},
  journal={arXiv preprint arXiv:2405.15668},
  year={2024}
}
```
