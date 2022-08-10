    title: Object Detection
    date: 2022.08.11
    category: paper
    tags:
    	- Object Detection
    	- paper
    toc: true
    author_profile: false
    sidebar:
      nav: "docs"


# A Survey of Modern Deep Learning based Object Detection Models

[Syed Sahil Abbas Zaidi](https://arxiv.org/search/cs?searchtype=author&query=Zaidi%2C+S+S+A), [Mohammad Samar Ansari](https://arxiv.org/search/cs?searchtype=author&query=Ansari%2C+M+S), [Asra Aslam](https://arxiv.org/search/cs?searchtype=author&query=Aslam%2C+A), [Nadia Kanwal](https://arxiv.org/search/cs?searchtype=author&query=Kanwal%2C+N), [Mamoona Asghar](https://arxiv.org/search/cs?searchtype=author&query=Asghar%2C+M), [Brian Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+B)

**[arXiv:2104.11892](https://arxiv.org/abs/2104.11892)**

![Screen Shot 2022-08-08 at 12.54.17 PM.png](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Screen_Shot_2022-08-08_at_12.54.17_PM.png)

이 논문은 총 9개의 section으로 구성되어 있으며, 각 section의 내용은 위와 같습니다.

II. BACKGROUND

A. Problem Statement

classification이 단순히 물체를 인지하는 것을 목표로 했다면, detection은 미리 정의된 클래스의 모든 instance를 감지하고 정렬된 상자에 의해 이미지에 대략적인 위치를 제공하는 것을 목표로합니다.

B. Key challenges in Object Detection

1. Intra class variation 

    조명, 자세, 시점 등 제약 없는 외부환경과 회전, 흐릿함, 주변환경 등으로 인해 구분이 어려운 경우에 대한 object detection 필요.

2. Number of categories 

    분류할 수 있는 객체 클래스가 너무 많은 경우 구분이 어려울 수 있고, 더 높은 품질의 lable이 필요.  이에 적은 sample을 잘 사용하는 방법에 대한 연구과제도 존재함.

3. Efficiency 

    많은 계산 리소스가 필요하지만, edge device의 보편화로, 더 효율적인 방법을 object detection 필요.


III. DATASETS ANDEVALUATIONMETRICS

A. Datasets

1. PASCAL VOC 07/12

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled.png)

    - PASCAL VOC는 PASCAL VOC challenge에서 쓰이던 데이터셋으로, 2005년에서 2012년까지 진행되었으며, 그 중 PASCAL 2007과 PASCAL 2012 데이터셋이 벤치마크 데이터셋으로 자주 사용됨.
    - 현재는 벤치마크용으로만 사용되며, 학습용으론 잘 쓰이지 않는 데이터셋
    - 클래스별 데이터수

        ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%201.png)

2. ILSVRC

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%202.png)

    - ILSVRC는 The ImageNet Large Scale Visual Recognition Challenge 에서 사용된 데이테셋으로, 2010년에서 2017까지 진행됨.
    - 클래스 별 데이터 수

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%203.png)

    - 위 두 데이터셋의 문제
        - 이미지 내 object가 큰 편임
        - object가 중앙에 잘 위치해 있음
        - 이미지당 object 수가 적음

3. MS-COCO

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%204.png)

    - COCO 데이터셋은 ImageNet 데이터셋의 문제점을 해결하기 위해 2014년에 제안되었으며, 다양한 크기의 물체가 존재하고, 높은 비율로 작은 물체들이 존재, 덜 Iconic함
    - Thing과 stuff를 구분.
        - Thing은 사람 차 개 등 레이블링이 쉬운 물체, stuff는 잔디, 하늘 등 레이블링이 어려운 물체들
    - 클래스별 데이터 수

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%205.png)

4. Open Image

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%206.png)

- google이 2017년에 구성한 데이터셋으로, 가장 크며, 이미지당 객체 수도 가장 많음.
- 클래스별 데이터 수

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%207.png)

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%208.png)

1. Issues of Data Skew/Bias

     데이터 종류에 따른 불균형이 심하다. Pascal VOC, MS-COCO, Open Images Dataset 의 경우 상위 5개 클래스 (사람, 자동차 등)의 경우 데이터가 많지만, 하위 항목의 경우 데이터가 부족.
     ​    
     심지어 MS-COCO의 헤어드라이어 이미지는 198개, Open Images Dataset의 Paper Cutter 이미지는 3개에 불과함. 때문에 모든 객체를 감지하는 모델을 생성 할 때 편향 존재.
     ​    
     ILSVRC의 경우 그나마 균형이 맞지만, 코알라, 컴퓨터 키보드 등 물체 감지 시나리오에서 많이 사용되지 않는 이미지가 많다는 단점 존재.

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%209.png)
​    

B. Metrics

1. IoU

박스의 겹치는 면적에 대한 지표 - detection 했을 때 생기는 박스와 실제 박스간 겹치는 정도에 대해 임계값 이상인 박스만 남기고, 해당 ground-truth로 분류

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2010.png)

1. mAP

    가장 일반적인 평가지표로, IoU가 임계값 보다 크면 True positive(물체를 잘 탐지함), 작으면 False Positive (물체가 없는데 탐지),  물체가 있는데 탐지 못함 (IoU 측정 불가) =  False Negative

    이에 대한 precision _ recall 그래프를 생성 하고, 그 아랫 부분의 넓이를 구한 값이 AP

    각 클래스당 AP를 구하여 평균을 취한 값이 mAP


IV. BACKBONE ARCHITECTURES

A. AlexNet

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2011.png)

- 최초의 CNN 기반 모델로 ReLU를 처음 사용하였다.
- Normalization Layer를 사용.
- Conv-Pool-Norm-Conv-Conv-Norm-Pool Layer
- Data Augmentation을 다수 사용.
- Learning rate는 0.01 이후 성능이 정체되면 1/10으로 줄인다.
- 11x11, 5x5 filter를 사용

B.VGG

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2012.png)

5 x 5 filter 1개를 사용하는 경우와 3 x 3 filter 2개를 사용하는 경우를 비교하면, 둘 다 5 x 5 범위

이에 VGGNet에서는 filter 크기를 3인 것만 사용하고, 대신 모델의 깊이를 더 깊게 쌓아서 더 좋은 성능 도출

- 1개의 5 x 5 filter는 총  1*5*5*채널 개의 parameter를 사용
- 2개의 3 x 3 filter는 총 2*3*3*채널 개의 parameter를 사용


C.GoogLeNet

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2013.png)

filter size를 1,3,5로 다양하게 사용하여 여러 receptive field size를 취하고 channel별로 concat하여 사용

(b)에서 처럼 계산량이 많은 3x3, 5x5 필터 적용 전 1x1를 적용하여 채널 수를 줄이면, 연산량이 감소

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2014.png)

빨간 부분은 앞선 두 backbone 구조 처럼 Conv-Pool-Norm-Conv-Conv-Norm-Pool 구조를 따르고, 노란 부분에서 inception module을 사용. 

그 과정에서 auxiliary classification loss를 사용하여, 역전파 시 gradient vanishing 해결.

최종 출력 inference 과정에서는 마지막 파란 부분의 softmax만을 사용

D.ResNets

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2015.png)

Microsoft에서 개발한 모델로 2015년 발표되었고, 최초로 사람의 분류 성능을 뛰어넘은 모델로 평가됨

shallow 모델이 deep 모델 보다 성능이 높게 나오는 현상으로부터 아이디어 구상

강제로 이미 학습한 부분을 다음 layer에 전달하면, deep 모델이 shallow 모델 만큼의 성능은 보장된다고 가정

이에 이전 결과를 다음 결과의 끝에 가산하는 skip connection 도입

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2016.png)

 실제로 Residual Networks Behave Like Ensembles of Relatively Shallow Networks 논문에 따르면, skip connection은 단순히 이전 결과를 반영(dependence와 중복 문제) 하는 것을 넘어 각 paths가 다른것에 강하게 종속되어있지 않아서 ensemble의 성격을 가진다고 주장

E.EfficientNet

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2017.png)

 모델의 크기를 키움으로써 성능을 높이는 3가지 방법 존재

1. network의 depth를 깊게 만드는 것
2. channel width(filter 개수)를 늘리는 것
3. input image의 해상도를 올리는 것

EfficientNet은 AutoML을 통해 이 3가지의 최적의 조합을 찾은 것

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2018.png)

with와, depth, resoution은 서로 긴밀히 연관되어있으며, 이들을 같이 키우는 것이 자원을 효율적으로 사용하며, 성능이 더 좋다는 것을 증명

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2019.png)

실제 다른 모델들과 efficentnet들을 비교한 결과, 대부분 기존 모델들보다 효율적이며, 8.4배 적은 연산량으로 더 높은 정확도를 갖는 모델도 존재

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2020.png)

평가지표 및 수식을 통한 검증 이외에 compound scaling method가 다른 방법에 비해 더 좋은 이유를 설명하는 Class activation map을 보면, 실제로 compound scaling을 사용했을 때 이미지의 주요 부분에 집중하고 있다는 것을 알 수 있다.

V. OBJECT DETECTORS

A. Pioneer Work

1. Viola-Jones

    Haar-like feature 사용, 

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2021.png)

2. HOG Detector

    Gradient를 구함

    Gradient를 이용해서 Cell크기 별로 histogram을 생성

    histogram의 Bin을 block크기로 나열

3. DPM

    딥 러닝 시대 이전의 가장 성공적인 알고리즘

    ‘For example, a human body can be considered as a collection of parts like head, arms, legs and torso. One model will be assigned to capture one of the parts in the whole image and the process is repeated for all such parts. A model then removes improbable configurations of the combination of these parts to produce detection.’

    전체 이미지를 부품의 조합으로 보고, 부품 하나씩을 포착, 후에 부품을 조합하여 있을 수 없는 조합을 제거하는 방식

- 기존 방식들은 gradient를 계산해서 histogram을 생성하고, bin을 나열 할 때 영상에 대한 위치 정보가 소실됨.

B. Two-Stage Detectors

1. R-CNN

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2022.png)

    1. selective search로 object가 있을법한 위치를 2000개 찾는다. (Region Proposal)
    2. Region Proposal과 Ground truth에 대한 IoU를 계산하여 RoI를 찾는다.
    3. RoI를 잘라내고, AlexNet의 인풋 size로 Warping
    4. Warping된 데이터를 pre-trained AlexNet에 적용하여 feature map 추출
    5. 추출된 feature map으로부터 각 class에 대한 linear svm을 적용하여 분류 학습
    6. 동일한 object중 score가 가장 높은 bounding box만 남기고 나머지는 제거
    7. 결과 box를 예측 box에 맞추기 위한 회귀 수행 

2. SPP-Net

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2023.png)

    1. 문제
        1. 기존 CNN Architecture의 고정된 입력 size에 맞게 이미지를 Crop/Warping하는 과정에서 왜곡 현상 발생
        2. Selective Search의 결과에 대한 최대 2000번의 순차적 CNN 연산 비효율성
    2. 해결방안
        1. Region Proposal과 CNN 순서 변경 (feature map에 대해 Region Proposal을 수행)
        2. 입력 이미지의 크기에 관계 없이 Conv layer들을 통과시키고, FC layer 통과 전에 피쳐 맵들을 동일한 크기로 조절해주는 pooling을 적용(Spatial Pyramid Pooling)

3. Fast R-CNN

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2024.png)

    1. 문제
        1. 3단계 Train 파이프라인 존재(SS, CNN, SVM)
        2. 4x4, 2x2, 1x1 spatial bin으로 인한 Overfitting
        3. SVM 학습으로 인한 대용량 저장 공간 필요
    2. 해결방안
        1. End-to-End model로 변경하여 전체 파라미터에 대한 loss를 학습
        2. SPP-Net에서 사용한 1x1, 2x2, 4x4 3가지 spatial bin 대신 7x7 spatial bin 하나를 사용
        3. softmax로 ROI Classification과 bbox Regression

4. Faster R-CNN

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2025.png)

    1. 문제
        1. Region Proposal dependency
    2. 해결방안
        1. Region Proposal을 GPU로 처리할 수 있는 RPN을 추가
            1. feature map에 anchor box를 적용하여
            2. 해당 box에 대해 classification, Bbox regression 수행

5. FPN

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2026.png)

    1. 작은 scale의 object를 탐지하는데 어려움
        1. 입력이미지 크기 변경
        2. feature map으로부터 object detection
    2. 방안 1,2는 각각 연산량 증가 및 정보소실 문제 발생
    3. 단계별로 비례한 크기의 feature map을 생성하고 가장 상위 layer에서부터 내려오면서 feauture map을 합쳐주는 방식 활용
        1. 상위 layer의 추상화된 정보와 하위 레이어의 작은 물체 정보를 모두 활용

6. DetectoRS

C. Single Stage Detectors

1. YOLO

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2027.png)

    1. 2-stage method(RPN+Classification)는 Real-time Detection에 적합하지 않은 처리 속도
    2. 전체 이미지에 대한 Large context를 파악, RPN과 Classification을 병렬적으로 처리하는 1-stage detector 생성
        1. 이미지를 여러 그리드로 분할
        2. 상자의 중심, 상자의 너비와 높이, 상자가 객체를 포함할 확률, 객체의 클래스를 나타내는 로스를 하나의 multi loss로 정의
        3. 셀마다 여러장의 바운딩 박스 생성
        4. Confidence Score(물체가 존재할 확률 **∗** IoU**)**가 높은 바운딩박스를 객체로 간주
2. SSD

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2028.png)

    1. 입력 이미지를 7x7 크기의 그리드로 나누고, 각 그리드 별로 Bounding Box Prediction을 진행하기 때문에 그리드 크기보다 작은 물체를 잡아내지 못함
    2. 정확도가 하락
        1. 단계별 피쳐맵에 대해 모두 Object Detection을 수행
        2. 높은 해상도의 feature map이 생성되면, 작은물체를 낮은 해상도의 feature map에서는 큰 물체를 detection 하여 각각의 feature map을 통해 생성한 bounding box에 대해 regression 및 classification 수행
3. YOLOv2 and YOLO9000
    1. 문제
        1. **Group of small objects** 
            1. 각 grid cell은 하나의 class만 예측 가능함
            2. Grid cell 하나보다 작은 크기의 Object 문제점
        2. **Unusal aspect ratios**
            1. Training data에 한해 Bbox를 학습하므로 새로운 형태의 Bbox를 예측하지 못함
    2. contribution

        ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2029.png)

4. RetinaNet
    1. Class imbalabce 문제를 다루기 위한 일반적인 방법은 weighting factor *α*를 사용하는 것
    2. positive/negitive example들의 균형을 맞춰주지만, easy/hard example들을 구별 못함
        1. **Focal Loss 정의**
        2. 입력 이미지의 밀도 샘플링으로 물체 예측
5. YOLOv3

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2030.png)

    1. Small object 또는 Overlapping된 Object에 대한 Localization error 발생
        1. 총 3개의 Scale을 사용하며, 각 scale 당 3개의 anchor box 생성
    2. multi-label classification for objects
6. CenterNet

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2031.png)

    1. 한 점 (center point)과 물체의 사이즈, offset을 output으로 하여 예측 object detection 뿐만 아니라 3D detection, pose estimation에서도 활용
7. EfficientDet

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2032.png)

    1. EfficientNet과 마찬가지로 compound scaling 방식을 사용
    2. multi-scale feature 합성방식인 BiFPN 제안
8. Swin  Transformer

    ![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2033.png)

9. 텍스트와는 다른 이미지만의 특성을 활용한 vision transformer
    1. 객체의 크기가 일정하지 않음
    2. 해상도가 존재하며, 큰 경우 계산 복잡도가 큼
10. local window를 모델에 적용
    1. 기존  ViT와 다르게 계층마다 다른 해상도의 결과를 얻을 수 있음
    2. 계산 복잡도 감소

VI. LIGHTWEIGHT NETWORKS

 생략

VII. COMPARATIVE RESULTS

![Untitled](A%20Survey%20of%20Modern%20Deep%20Learning%20based%20Object%20Dete%20ce03c02f289d4a379f2e6ead6ab0fbbb/Untitled%2034.png)

VIII. FUTURE TRENDS

1. AutoML : object detector의 특성을 결정하기 위한 NAS의 발전 ( Efficient net)
2. Lightweight detectors :경량 네트워크들의 정확도 향상
3. Weakly supervised/few shot detection : 작은 정보로 라벨링된 데이터에 대한 성능 향상
4. Domain transfer : 특정 도메인에서 특정 데이터를 사용할 수 있도록 훈련된 모델의 재사용 및 데이터셋의 가용성 향상
5. 3D object detection : 사람 수준 이상의 3D 물체 감지 향상 (자율주행)
6. Object detection in video : 객체 인식을 위한 프레임간의 공간적 시간적 관계 사용 향상

IX. CONCLUSION

SwinTransformer를 사용해야한다.