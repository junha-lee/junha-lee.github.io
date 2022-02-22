---
title: capture & predict (raspberrypi)
date: 2020-08-12
category: project
tags:
    - image detection
    - coal
    - IoT
toc: true
author_profile: false
sidebar:
  nav: "docs"
---





### Raspberry pi를 이용한 석탄 분류 장치 만들기

---


```python
def capture(camid = CAM_ID): #사진 찍어서 저장
    cam = cv2.VideoCapture(camid)
    if cam.isOpened() == False:
        print ('cant open the cam (%d)' % camid)
        return None
    ret, frame = cam.read()
    if frame is None:
        print ('frame is not exist')
        return None
    cv2.imwrite('coal.png',frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
    cam.release()
def load_image(filename): # 저장한 이미지 불러와 형식 지정
    img = load_img(filename, target_size=(150,150))
    img = img_to_array(img)
    img = img.reshape(1,150,150, 3)
    img = img.astype(＇float32＇)
    return img
def predict(): # 저장한 모델 불러오기 후 load_image 호출 하여 예측 
    model = tf.keras.models.load_model('coal_classification_data_augmentation - epoch17.h5')
    img = load_image('./coal.png')
    result = model.predict(img)
    if (result[0]==1):
        return '유연탄'
    elif (result[0]==0):
        return '연갈탄'
if __name__ == '__main__':
    capture()
    print(predict())
```

 가장 성능이 좋았던 "coal_classification_data_augmentation-epoch17" 모델을 사용했습니다.

<iframe width="560" height="315" src="https://www.youtube.com/embed/ogs_l6xo1nE" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


### 마치며

raspberry pi 석탄 분류기를 만들어봤습니다.
이제 여러 종류의 석탄을 탐지하여 실제 컨테이너 밸트에서 분류하는 장치를 만들어 볼 생각입니다.

읽어주셔서 감사합니다.
