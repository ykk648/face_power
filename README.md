## Intro

Separate from [AI_power](https://github.com/ykk648/AI_power), consists methods/onnx-models for face related missions.

Examples can be found in [examples](./examples).

Models can be found in [huggingface](https://huggingface.co/ykk648/face_lib).

---

### Eye Detect

- eye open detect from [Open-eye-closed-eye-classification](https://github.com/abhilb/Open-eye-closed-eye-classification)
- iris detect from [iris-detection](https://github.com/Kazuhito00/iris-detection-using-py-mediapipe)


### Face 3D

- [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
- [3dRecon](https://modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/summary)


### Face Attribution

- based on [mmclassification](https://github.com/open-mmlab/mmclassification), already convert to onnx.
- supply pretrained model trained by private dataset, mind the face should be aligned first.

### Face Alignment

- ffhq align method
- [face alignment](https://github.com/1adrianb/face-alignment) from 1adrianb
- conform multi similarity align methods

### Face Detect

- mtcnn from [mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch)
- scrfd from [insightface](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- s3fd from [DeepFaceLab](https://github.com/iperov/DeepFaceLab)

### Face Embedding

- Arcface from [SimSwap](https://github.com/neuralchen/SimSwap) (mod to jit model)
- [CurricularFace](https://github.com/HuangYG123/CurricularFace) (mod to jit model)

### Face Landmark

- pfpld from [nniefacelib](https://github.com/hanson-young/nniefacelib/tree/master/PFPLD)

### Face Parsing

- [face-parsing.PyTorch](./face_parsing/face_parsing_api.py), onnx converted
- [DFL Xseg](./face_parsing/dfl_xseg_net.py), based on opensource, self trained (private data)

### Face Restore 

- [Gpen](https://github.com/yangxy/GPEN)
- [DFDNet](https://github.com/csxmli2016/DFDNet) (add batch parallel support)
- [GFPGAN](https://github.com/TencentARC/GFPGAN) onnx converted
- [RestoreFormer](https://github.com/wzhouxiff/RestoreFormer)
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- ESRGAN etc. TODO

### Face Swap

- [FaceFusion](https://modelscope.cn/models/damo/cv_unet-image-face-fusion_damo/)
- [Hififace](https://johann.wang/HifiFace/)
- [InSwapper](https://github.com/deepinsight/insightface)


### Head Pose

- [WHENet](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet)

---

## Related Repo
- [AI_power](https://github.com/ykk648/AI_power)
- [cv2box](https://github.com/ykk648/cv2box)
- [apstone](https://github.com/ykk648/apstone)