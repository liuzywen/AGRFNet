# AGRFNet: Two-stage Cross-modal and Multi-level Attention Gated Recurrent Fusion Network for RGB-D Saliency Detection
The paper has been published in Signal Processing Image Communication.
The detail can be seen in [paper](https://github.com/liuzywen/AGRFNet/blob/main/AGRFNet.pdf)


# Abstract: 
RGB-D saliency detection aims to identify the most attractive objects in a pair of color and depth images. However, most existing models adopt classic U-Net framework which progressively decodes two-stream features. In this paper, we decode the cross-modal and multi-level features in a unified unit, named Attention
Gated Recurrent Unit (AGRU). It can reduce the influence of low-quality depth image, and retain more semantic features in the progressive fusion process. Specifically, the features of different modalities and different levels are organized as the sequential input, recurrently fed into AGRU which consists of reset gate, update gate and memory unit to be selectively fused and adaptively memorized based on attention mechanism. Further, two-stage AGRU serves as the decoder of RGB-D salient object detection network, named AGRFNet. Due to the recurrent nature, it achieves the best performance with the little parameters. In order to further improve the performance, three auxiliary modules are designed to better fuse semantic information, refine the features of the shallow layer and enhance the local detail. Extensive experiments on seven widely used benchmark datasets demonstrate that AGRFNet performs favorably against 18 state-of-the-art RGB-D SOD approaches.

## Pretraining 

链接：https://pan.baidu.com/s/1yf94REEnKRHqkmmSla5jPg 
提取码：wsbg 




## Training Set
2185
https://drive.google.com/file/d/1fcJj4aYdJ6N-TvvxSZ_sBo-xhtd_w-eJ/view?usp=sharing


2985
https://drive.google.com/file/d/1mYjaT_FTlY4atd-c0WdQ-0beZIpf8fgh/view?usp=sharing

##  Result Saliency Maps
链接：https://pan.baidu.com/s/1deM1TDaWtx-4eWutyfoaPw 
提取码：y3q0 





### Citation

If you find the information useful, please consider citing:

```
@article{liu2022agrfnet,
  title={AGRFNet: Two-stage cross-modal and multi-level attention gated recurrent fusion network for RGB-D saliency detection},
  author={Liu, Zhengyi and Wang, Yuan and Tan, Yacheng and Li, Wei and Xiao, Yun},
  journal={Signal Processing: Image Communication},
  pages={116674},
  year={2022},
  publisher={Elsevier}
}
```
