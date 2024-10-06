# WDYS
PyTorch implementation of the paper: 

[What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Vision-Language Models](https://arxiv.org/abs/2405.15668). 

[Abdelrahman Abdelhamed](https://abdokamel.github.io/)<sup>*1</sup>, [Mahmoud Afifi](https://www.mafifi.info/)<sup>*2</sup>, [Alec Go](https://scholar.google.com/citations?user=I54RNh4AAAAJ&hl=en&oi=sra)<sup>1</sup>

<sup>*</sup>*Equal contribution*

<sup>1</sup>Google Research  &ensp; <sup>2</sup>Google




<p align="center">
  <img src="https://github.com/user-attachments/assets/d9743225-29f6-40ef-b1f7-f3f1be5af401" width="40%">
</p>



**Abstract**

Large language models (LLMs) has been effectively used for many computer vision tasks, including image classification. In this paper, we present a simple yet effective approach for zero-shot image classification using multimodal LLMs. By employing multimodal LLMs, we generate comprehensive textual representations from input images. These textual representations are then utilized to generate fixed-dimensional features in a cross-modal embedding space. Subsequently, these features are fused together to perform zero-shot classification using a linear classifier. Our method does not require prompt engineering for each dataset; instead, we use a single, straightforward, set of prompts across all datasets. We evaluated our method on several datasets, and our results demonstrate its remarkable effectiveness, surpassing benchmark accuracy on multiple datasets. On average over ten benchmarks, our method achieved an accuracy gain of 4.1 percentage points, with an increase of 6.8 percentage points on the ImageNet dataset, compared to prior methods. Our findings highlight the potential of multimodal LLMs to enhance computer vision tasks such as zero-shot image classification, offering a significant improvement over traditional methods.


**Code**

Note that this implementation is unofficial and provided for research and experimental purposes.

TBE

If you use this code, please cite our paper:
```
@article{abdelhamed2024WDYS,
  title={What Do You See? Enhancing Zero-Shot Image Classification with Multimodal Large Language Models},
  author={Abdelhamed, Abdelrahman and Afifi, Mahmoud and Go, Alec},
  journal={arXiv preprint arXiv:2405.15668},
  year={2024}
}
```
