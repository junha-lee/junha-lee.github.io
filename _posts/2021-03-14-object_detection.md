---
title: Object Detection
date: 2020-09-01
category: coal
toc: true
---



이번 포스트에서는 라즈베리파이를 이용해 석탄을 탐지하기 위해 object Detection 논문들을 리뷰합니다.

### Object Detection
---

object detection은 classification + localization 으로 여러가지 object에 대한 classification과 그 object들의 위치정보를 파악하는 것을 동시에 하는 분야

![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/object-detection-1.png)

위와 같은 알고리즘들이 있으며 localization과 classification이 따로 이루어 지면 2-Stage Detector, 동시에 이루어지면 1-Stage Detector이고, anchor box를 사용하는지 여부에 따라 나눌 수 있다.

![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/object-detection.png)

### Faster R-CNN
---

![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/fasterr-cnn.png)

1. Convolution layer 적용하여 feature map 생성
2. RoI(object가 있을 것 같은 영역) 생성
3. RoI 사이즈 통일 후 분류하여 이미지 라벨과 bounding box 추출

이전과 다른 점
RPN 딥러닝 네트워크를 이용하여 RoI 생성

*** RPN : feature map에 sliding window를 적용시켜 각 window마다 k개의 anchor box들을 이용해 그 영역에 object가 있는지/없는지 스코어와 4개의 좌표를 뽑아내는 것


### YOLO
---
![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo.png)

input 이미지를 SxS grid로 나누고 각 grid 영역에 해당하는 박스 및 가능성 예측 별도의 좌표 예측없이 fully connected layer를 거치면 bounding box별 class probability를 바로 예측한다.

Thresh를 조절하여 정확도별 물체를 표현할 수 있다.

### SSD
---
![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/ssd.png)

이미지에서 보여지는 object들의 크기는 매우 다양하고  때문에 Yolo와 같이 convolution layer 들을 통해 추출한 한 feature map으로 detect 하기에는 부족하다.

다양한 위치의 layer들에서 image feature를 추출하여 detector와 classifier를 적용

앞쪽 layer에서는 receptive field size가 작으니 더 작은 영역을 detect 할 수 있고, 뒤쪽 layer로 갈수록 receptive field size가 커지므로 더 큰 영역을 detect 할 수 있다는 특징을 이용

### CornerNet
---
![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/cornernet.png)

이미지에 1~2개의 물체를 detect 하기위해 수백 수천개의 anchor box를 사용하는 것은 낭비이고, 어떤 모양의 anchor box를 사용할 것인가(hyperparameter setting)도 큰 문제.

hourglass 네트워크를 통해 두 개의 포인트 (왼쪽 위 포인트, 오른쪽 아래 포인트)에 해당하는 값을 얻고, Prediction module을 통해 그 위치의 heatmap, 같은 물체인지 파악할 수 있는 embedding, 정확한 bounding box 위치 파악을 위한 offset을 얻고 이것들을 이용해 detect

### CenterNet
---

![object_detect](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/centernet.png)

CornerNet이 2개의 점을 추출하였다면, CenterNet은 오직 single point로 object를 detect 

한 점 (center point)과 물체의 사이즈, offset을 output으로 하여 예측 object detection 뿐만 아니라 3D detection, pose estimation에서도 활용


### Reference
---

* Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
* You Only Look Once: Unified, Real-Time Object Detection
* SSD: Single Shot MultiBox Detector
* CornerNet: Detecting Objects as Paired Keypoints
* CenterNet: Keypoint Triplets for Object Detection
* https://nuggy875.tistory.com/20
* https://blog.lunit.io/2017/06/01/r-cnns-tutorial/
* https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html
* https://nuguziii.github.io/survey/S-001/


```python

```
