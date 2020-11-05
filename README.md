# docker images method notes
- video demo has added, You need to do these things:
  - use the new Dockerfile to build image
    - python lib version all specified
  - run and within the container
    - docker run --gpus all -v DensePose/DensePoseData:/denseposedata -v DensePose/tools:/densepose/tools_kk -it densepose:c2-cuda9-cudnn7 bash
  - replace the local DensePoseData directory with the host one
    - mv /densepose/DensePoseData /densepose/DensePoseDataLocal
    - ln -s /denseposedata DensePoseData
  - copy tools_kk/vis_kk.py to detectron/utils(you can also do it in infer_vid.py when running demo)
  - install ffmpeg(you can also do it in dockerfile when building the image)
  - save container to new image
    - docker commit $(docker ps --last 1 -q) 
 densepose:c2-cuda9-cudnn7-kk

- video demo command:
  - docker run --rm --gpus all -v /home/kevin/aj/DensePose/DensePoseData:/denseposedata -v /home/kevin/aj/DensePose/tools:/densepose/tools_kk -it densepose:c2-cuda9-cudnn7-kk python2 tools_kk/infer_vid.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/infer_out/  --wts ./DensePoseData/DensePose_ResNet101_FPN_s1x-e2e.pkl --input-file tools_kk/video.mp4

- image demo command:
  - docker run --rm --gpus all -v /home/kevin/aj/DensePose/DensePoseData:/denseposedata -v /home/kevin/aj/DensePose/tools:/densepose/tools_kk -it densepose:c2-cuda9-cudnn7-kk python2 tools_kk/infer_simple.py --cfg configs/DensePose_ResNet101_FPN_s1x-e2e.yaml --output-dir DensePoseData/infer_out/ --image-ext jpg --wts ./DensePoseData/DensePose_ResNet101_FPN_s1x-e2e.pkl DensePoseData/demo_data/grc.jpg

- reference
  - https://github.com/trrahul/densepose-video


# DensePose: 
**Dense Human Pose Estimation In The Wild**

_Rıza Alp Güler, Natalia Neverova, Iasonas Kokkinos_

[[`densepose.org`](https://densepose.org)] [[`arXiv`](https://arxiv.org/abs/1802.00434)] [[`BibTeX`](#CitingDensePose)]

Dense human pose estimation aims at mapping all human pixels of an RGB image to the 3D surface of the human body. 
DensePose-RCNN is implemented in the [Detectron](https://github.com/facebookresearch/Detectron) framework and is powered by [Caffe2](https://github.com/caffe2/caffe2).

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1qfSOkpueo1kVZbXOuQJJhyagKjMgepsz" width="700px" />
</div>


In this repository, we provide the code to train and evaluate DensePose-RCNN. We also provide notebooks to visualize the collected DensePose-COCO dataset and show the correspondences to the SMPL model.

## Installation

Please find installation instructions for Caffe2 and DensePose in [`INSTALL.md`](INSTALL.md), a document based on the [Detectron](https://github.com/facebookresearch/Detectron) installation instructions.

## Inference-Training-Testing

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md)  for examples of inference and training and testing.

## Notebooks

### Visualization of DensePose-COCO annotations:

See [`notebooks/DensePose-COCO-Visualize.ipynb`](notebooks/DensePose-COCO-Visualize.ipynb) to visualize the DensePose-COCO annotations on the images:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1uYRJkIA24KkJU2i4sMwrKa61P0xtZzHk" width="800px" />
</div>

---

### DensePose-COCO in 3D:

See [`notebooks/DensePose-COCO-on-SMPL.ipynb`](notebooks/DensePose-COCO-on-SMPL.ipynb) to localize the DensePose-COCO annotations on the 3D template ([`SMPL`](http://smpl.is.tue.mpg.de)) model:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1m32oyMuE7AZd3EOf9k8zHpr75C8bHlYj" width="500px" />
</div>

---
### Visualize DensePose-RCNN Results:

See [`notebooks/DensePose-RCNN-Visualize-Results.ipynb`](notebooks/DensePose-RCNN-Visualize-Results.ipynb) to visualize the inferred DensePose-RCNN Results.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1k4HtoXpbDV9MhuyhaVcxDrXnyP_NX896" width="900px" />
</div>

---
### DensePose-RCNN Texture Transfer:

See [`notebooks/DensePose-RCNN-Texture-Transfer.ipynb`](notebooks/DensePose-RCNN-Texture-Transfer.ipynb) to localize the DensePose-COCO annotations on the 3D template ([`SMPL`](http://smpl.is.tue.mpg.de)) model:

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1r-w1oDkDHYnc1vYMbpXcYBVD1-V3B4Le" width="900px" />
</div>

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

## <a name="CitingDensePose"></a>Citing DensePose

If you use Densepose, please use the following BibTeX entry.

```
  @InProceedings{Guler2018DensePose,
  title={DensePose: Dense Human Pose Estimation In The Wild},
  author={R\{i}za Alp G\"uler, Natalia Neverova, Iasonas Kokkinos},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2018}
  }
```


