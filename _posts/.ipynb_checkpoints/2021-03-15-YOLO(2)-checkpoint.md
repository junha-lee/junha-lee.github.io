---
title: YOLO(2)
date: 2020-10-2
category: coal
toc: true
---

이번 포스트에서는 라즈베리파이를 이용해 석탄을 탐지하기 위한 coal 데이터 전처리 과정과 YOLO를 다룹니다.

### Yolo 데이터 전처리
---

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-data-preprocessing.png)

YOLO 학습을 위해서 classification에서 사용한 coal 데이터에 실측상자를 생성하고, 레이블을 달아줍니다.

### Darkflow 매개변수
---

--imgdir&nbsp;&nbsp;&nbsp;테스트 이미지 경로

--binary&nbsp;&nbsp;&nbsp;weights 경로

--config &nbsp;&nbsp;&nbsp;cfg경로

--dataset       &nbsp;&nbsp;&nbsp;dataset 경로

--labels      &nbsp;&nbsp;&nbsp;  labels file경로

--backup      &nbsp;&nbsp;&nbsp;  backup folder 경로

--summary &nbsp;&nbsp;&nbsp;      TensorBoard summaries 

--annotation &nbsp;&nbsp;&nbsp;	박스 경로

--threshold  &nbsp;&nbsp;&nbsp;   detection threshold

--model 	&nbsp;&nbsp;&nbsp;    모델 경로

--momentum     &nbsp;&nbsp;&nbsp; momentum optimizers

--save	  &nbsp;&nbsp;&nbsp;      체크포인트 저장

--train       &nbsp;&nbsp;&nbsp;  train the whole net

--savepb 	&nbsp;&nbsp;&nbsp;	가중치 저장

--gpu      &nbsp;&nbsp;&nbsp;     GPU 사용량

--gpuName   &nbsp;&nbsp;&nbsp;    GPU device name

--lr     &nbsp;&nbsp;&nbsp;       learning rate

--keep 	  &nbsp;&nbsp;&nbsp;      저장할 최근 학습 결과

--batch    &nbsp;&nbsp;&nbsp;     batch size

--epoch     &nbsp;&nbsp;&nbsp;    number of epoch

--demo     &nbsp;&nbsp;&nbsp;     demo on webcam

--queue    &nbsp;&nbsp;&nbsp;     process demo in batch

--json 	  &nbsp;&nbsp;&nbsp;      json 출력

--saveVideo   &nbsp;&nbsp;&nbsp;  생성된 비디오 출력

### YOLO 학습
---

python flow --model ./cfg/tiny-yolo.cfg --labels ./labels.txt --trainer adam --dataset ./detect/image/ --annotation ./detect/annotation/ --train --summary ./logs --batch 4 --epoch 200 --save 50 --keep 5 --lr 1e-04 --gpu 0.5

모델,라벨,데이터,박스데이터 경로 불러와서 -> adam사용하고 batch=4, epoch는 200번 학습률은 0.0001로 GPU를 반만 사용하여 학습 한다. 그 과정에서 50개의 배치마다 저장, 마지막 5개의 모델 저장하며 TensorBoard summaries를 사용한다.


### 부가 설명
---
* 모델 사용

-> python flow --imgdir ./detect_testset -–model ./cfg/tiny-yolo.cfg --load -1 --batch 1 --threshold 0.5

* 그래프 (Tensor Board)

-> tensorboard --logdir=./logstrain/ 


### Darkflow 에러 해결 방법

---
Darkflow.sln 이나 no_gpu_darkflow.sln 을 빌드 불가

비주얼 스튜디오 2015를 설치 후 환경변수 편집


2. GPU 사용시 darkflow.exe 파일 실행 

CUDA 8.0 사용 및 Cudnn 버전 맞춤


3. CUDA_ERROR_OUT_OF_MEMORY, CUDNN_STATUS_ALLOC_FAILED

(--gpu ? --batch ?) gpu 사용량, batch 크기 조정 or 모델 구조 변경


4. AssertionError: expect 202335260 bytes, found 203934260

 - darkflow\utils\  폴더의  loader.py에서
self.offset = 16을  찾아 
self.offset = 16 + found_value - expected_value 형식으로 변경
Ex)self.offset = 16 + 203934260 - 202335260 


### train result
---

Train dataset : 이미지 파일과 박스 전처리 한 xml 파일 29개

--> Batch를 4로 설정 : 한 과정당 4개의 데이터 필요

--> 한 에포크는 7개의 학습 과정으로 구성 (4*7=28)    

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-result1.png)

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-result2.png)

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-result3.png)

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-result-1.png)

![YOLO](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/yolo-result-2.png)

testset에 대해서는 한 유연탄에 대해 90% 이상의 정확도로 탐지하지만, 나머지는 탐지하지 못했고,

trainset에 대해서는 모든 석탄을 탐지함을 알 수 있었습니다.

### 마치며
---
trainset에 대해서는 모든 석탄을 탐지함을 알 수 있었지만, testset에 대해서는 일부만 탐지가 가능한 것으로 보아 일부 데이터에 과적합 되었다고 추측할 수 있습니다.

특히 연구에 사용한 모델 구조와 하이퍼파라미터 튜닝값을 Demo data에 사용한 결과 testset까지 모두 탐지하는 것으로 보아 Imagenet과 같은 다양한 종류의 방대한 데이터가 필요하다고 생각합니다.



