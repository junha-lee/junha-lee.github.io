```python
import os
train_dir = './input/dataset2/train'
validation_dir ='./input/dataset2/test'

train_1_dir = os.path.join(train_dir, '연갈탄')

train_2_dir = os.path.join(train_dir, '유연탄')

validation_1_dir = os.path.join(validation_dir, '연갈탄')

validation_2_dir = os.path.join(validation_dir, '유연탄')
```


```python
train_1_fnames = os.listdir(train_1_dir)
print(train_1_fnames[:100])

train_2_fnames = os.listdir(train_2_dir)
train_2_fnames.sort()
print(train_2_fnames[:100])
```

    ['0.jpg', '1.jpg', '11.jpg', '12.jpg', '14.jpg', '15.jpg', '17.jpg', '18.jpg', '20.jpg', '21.jpg', '22.jpg', '23.jpg', '24.jpg', '26.jpg', '27.jpg', '3.jpg', '5.jpg', '6.jpg', '7.jpg', '9.jpg', '_0_1000.png', '_0_1017.png', '_0_1026.png', '_0_1034.png', '_0_1061.png', '_0_1065.png', '_0_1136.png', '_0_1159.png', '_0_118.png', '_0_1195.png', '_0_1199.png', '_0_1224.png', '_0_1245.png', '_0_1283.png', '_0_1285.png', '_0_1309.png', '_0_1329.png', '_0_1341.png', '_0_1351.png', '_0_1362.png', '_0_1413.png', '_0_1421.png', '_0_1449.png', '_0_145.png', '_0_1460.png', '_0_1473.png', '_0_1501.png', '_0_1510.png', '_0_1522.png', '_0_1528.png', '_0_1529.png', '_0_156.png', '_0_1594.png', '_0_1655.png', '_0_1670.png', '_0_1674.png', '_0_1701.png', '_0_1712.png', '_0_1721.png', '_0_1734.png', '_0_1781.png', '_0_1805.png', '_0_181.png', '_0_1824.png', '_0_1830.png', '_0_1859.png', '_0_1903.png', '_0_1960.png', '_0_2010.png', '_0_2042.png', '_0_205.png', '_0_2052.png', '_0_2071.png', '_0_2084.png', '_0_2134.png', '_0_2137.png', '_0_2151.png', '_0_2170.png', '_0_221.png', '_0_2246.png', '_0_2274.png', '_0_2279.png', '_0_2293.png', '_0_2303.png', '_0_2312.png', '_0_2316.png', '_0_2325.png', '_0_2341.png', '_0_2359.png', '_0_2367.png', '_0_2375.png', '_0_2385.png', '_0_2400.png', '_0_2412.png', '_0_2415.png', '_0_2429.png', '_0_2515.png', '_0_2589.png', '_0_2590.png', '_0_2634.png']
    ['10.png', '100.png', '101.png', '102.png', '104.png', '105.png', '107.png', '108.png', '109.png', '110.png', '112.png', '113.png', '114.png', '116.png', '117.png', '118.png', '120.png', '122.png', '123.png', '124.png', '129.png', '13.png', '131.png', '133.png', '134.png', '135.png', '136.png', '14.png', '140.png', '142.png', '145.png', '146.png', '147.png', '148.png', '149.png', '155.png', '157.png', '158.png', '159.png', '16.png', '167.png', '168.png', '17.png', '170.png', '172.png', '173.png', '174.png', '176.png', '178.png', '179.png', '18.png', '182.png', '184.png', '185.png', '188.png', '191.png', '192.png', '193.png', '194.png', '196.png', '2.png', '20.png', '22.png', '23.png', '24.png', '28.png', '29.png', '3.png', '30.png', '31.png', '33.png', '4.png', '40.png', '41.png', '44.png', '45.png', '46.png', '47.png', '5.png', '52.png', '53.png', '54.png', '55.png', '56.png', '57.png', '59.png', '6.png', '60.png', '61.png', '62.png', '64.png', '65.png', '66.png', '67.png', '68.png', '69.png', '7.png', '70.png', '73.png', '75.png']
    


```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
nrows = 4
ncols = 4
pic_index = 0
```


```python
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_1_pix = [os.path.join(train_1_dir, fname) 
                for fname in train_1_fnames[pic_index-8:pic_index]]
next_2_pix = [os.path.join(train_2_dir, fname) 
                for fname in train_2_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_1_pix+next_2_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)
  img = mpimg.imread(img_path)
  plt.imshow(img)
plt.show()
```


    
![png](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/coal_classification_data_augmentation_files/coal_classification_data_augmentation_3_0.png)
    



```python
from tensorflow.keras import layers
from tensorflow.keras import Model
```


```python

img_input = layers.Input(shape=(150, 150, 3))


x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)


x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)

x = layers.Dense(512, activation='relu')(x)

output = layers.Dense(1, activation='sigmoid')(x)

model = Model(img_input, output)
```


```python
model.summary()
```

    Model: "functional_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 150, 150, 3)]     0         
    _________________________________________________________________
    conv2d (Conv2D)              (None, 148, 148, 16)      448       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 17, 17, 64)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 18496)             0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               9470464   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 9,494,561
    Trainable params: 9,494,561
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss="binary_crossentropy", optimizer="nadam",metrics=['acc'])
```

<h1>데이터 부풀리기


```python
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
# next_1_pix = [os.path.join(train_1_dir, fname) 
#                 for fname in train_1_fnames]
# for j in next_1_pix:
#     img = load_img(j)
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
    
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir='./input/dataset1/변형연갈탄'):
#         i += 1
#         if i > 20:
#             break  

```


```python
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(
#         rotation_range=40,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True,
#         fill_mode='nearest')
# next_2_pix = [os.path.join(train_2_dir, fname) 
#                 for fname in train_2_fnames]
# for j in next_2_pix:
#     img = load_img(j)
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
    
#     i = 0
#     for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir='./input/dataset1/변형유연탄'):
#         i += 1
#         if i > 20:
#             break  

```


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(150, 150),  
        batch_size=32,
        
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
```

    Found 2791 images belonging to 2 classes.
    Found 20 images belonging to 2 classes.
    


```python
history=model.fit(train_generator,
epochs = 30,
validation_data = validation_generator
)
```

    Epoch 1/30
     1/88 [..............................] - ETA: 0s - loss: 0.7294 - acc: 0.2188WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0080s vs `on_train_batch_end` time: 0.0130s). Check your callbacks.
    88/88 [==============================] - 20s 228ms/step - loss: 0.2596 - acc: 0.8993 - val_loss: 0.2644 - val_acc: 0.8500
    Epoch 2/30
    88/88 [==============================] - 20s 230ms/step - loss: 0.1463 - acc: 0.9427 - val_loss: 0.0314 - val_acc: 1.0000
    Epoch 3/30
    88/88 [==============================] - 20s 230ms/step - loss: 0.1119 - acc: 0.9660 - val_loss: 0.0801 - val_acc: 1.0000
    Epoch 4/30
    88/88 [==============================] - 20s 227ms/step - loss: 0.0677 - acc: 0.9742 - val_loss: 0.0038 - val_acc: 1.0000
    Epoch 5/30
    88/88 [==============================] - 20s 228ms/step - loss: 0.0318 - acc: 0.9900 - val_loss: 0.0085 - val_acc: 1.0000
    Epoch 6/30
    88/88 [==============================] - 21s 240ms/step - loss: 0.0540 - acc: 0.9839 - val_loss: 0.0192 - val_acc: 1.0000
    Epoch 7/30
    88/88 [==============================] - 21s 236ms/step - loss: 0.0182 - acc: 0.9925 - val_loss: 0.0116 - val_acc: 1.0000
    Epoch 8/30
    88/88 [==============================] - 20s 227ms/step - loss: 0.0636 - acc: 0.9832 - val_loss: 0.0034 - val_acc: 1.0000
    Epoch 9/30
    88/88 [==============================] - 20s 226ms/step - loss: 0.0302 - acc: 0.9896 - val_loss: 0.0073 - val_acc: 1.0000
    Epoch 10/30
    88/88 [==============================] - 20s 228ms/step - loss: 0.0095 - acc: 0.9961 - val_loss: 0.0030 - val_acc: 1.0000
    Epoch 11/30
    88/88 [==============================] - 20s 225ms/step - loss: 0.0043 - acc: 0.9989 - val_loss: 0.0111 - val_acc: 1.0000
    Epoch 12/30
    88/88 [==============================] - 20s 227ms/step - loss: 0.0039 - acc: 0.9989 - val_loss: 0.0022 - val_acc: 1.0000
    Epoch 13/30
    88/88 [==============================] - 20s 232ms/step - loss: 0.0680 - acc: 0.9778 - val_loss: 0.0129 - val_acc: 1.0000
    Epoch 14/30
    88/88 [==============================] - 20s 230ms/step - loss: 0.0160 - acc: 0.9943 - val_loss: 6.5788e-04 - val_acc: 1.0000
    Epoch 15/30
    88/88 [==============================] - 20s 228ms/step - loss: 0.0018 - acc: 0.9996 - val_loss: 3.3615e-05 - val_acc: 1.0000
    Epoch 16/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.1351 - acc: 0.9624 - val_loss: 0.3433 - val_acc: 0.9000
    Epoch 17/30
    88/88 [==============================] - 20s 230ms/step - loss: 0.0397 - acc: 0.9846 - val_loss: 0.0444 - val_acc: 0.9500
    Epoch 18/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.0238 - acc: 0.9914 - val_loss: 0.0023 - val_acc: 1.0000
    Epoch 19/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.0189 - acc: 0.9932 - val_loss: 0.1167 - val_acc: 0.9500
    Epoch 20/30
    88/88 [==============================] - 21s 237ms/step - loss: 0.0085 - acc: 0.9971 - val_loss: 0.1143 - val_acc: 0.9500
    Epoch 21/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.0253 - acc: 0.9925 - val_loss: 0.0094 - val_acc: 1.0000
    Epoch 22/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.0053 - acc: 0.9979 - val_loss: 0.0014 - val_acc: 1.0000
    Epoch 23/30
    88/88 [==============================] - 20s 231ms/step - loss: 0.0016 - acc: 0.9996 - val_loss: 0.0140 - val_acc: 1.0000
    Epoch 24/30
    88/88 [==============================] - 20s 227ms/step - loss: 0.0026 - acc: 0.9989 - val_loss: 0.0055 - val_acc: 1.0000
    Epoch 25/30
    88/88 [==============================] - 20s 229ms/step - loss: 0.0075 - acc: 0.9979 - val_loss: 0.0251 - val_acc: 1.0000
    Epoch 26/30
    88/88 [==============================] - 20s 226ms/step - loss: 0.0022 - acc: 0.9993 - val_loss: 0.0820 - val_acc: 0.9500
    Epoch 27/30
    88/88 [==============================] - 20s 225ms/step - loss: 0.2178 - acc: 0.9423 - val_loss: 0.0437 - val_acc: 1.0000
    Epoch 28/30
    88/88 [==============================] - 20s 225ms/step - loss: 0.0610 - acc: 0.9731 - val_loss: 0.0133 - val_acc: 1.0000
    Epoch 29/30
    88/88 [==============================] - 20s 226ms/step - loss: 0.0261 - acc: 0.9910 - val_loss: 0.0060 - val_acc: 1.0000
    Epoch 30/30
    88/88 [==============================] - 20s 225ms/step - loss: 0.0289 - acc: 0.9914 - val_loss: 0.1990 - val_acc: 0.9000
    


```python
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) 
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.figure()


plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.figure()


```




    <Figure size 432x288 with 0 Axes>




    
![png](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/coal_classification_data_augmentation_files/coal_classification_data_augmentation_13_1.png)
    



    
![png](https://raw.githubusercontent.com/junha-lee/junha-lee.github.io/main/assets/images/coal_classification_data_augmentation_files/coal_classification_data_augmentation_13_2.png)
    



    <Figure size 432x288 with 0 Axes>



```python

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model


def load_image(filename):
	img = load_img(filename, target_size=(150,150))
	img = img_to_array(img)
	img = img.reshape(1,150,150, 3)
	img = img.astype('float32')
	return img

test_dir ='./input/dataset2/valid'
test_1_dir = os.path.join(test_dir, '연갈탄')
test_2_dir = os.path.join(test_dir, '유연탄')

test_1_fnames = os.listdir(test_1_dir)
test_2_fnames = os.listdir(test_2_dir)

test_1_pix = [os.path.join(test_1_dir, fname) 
                for fname in test_1_fnames]
test_2_pix = [os.path.join(test_2_dir, fname) 
                for fname in test_2_fnames]
p1=0

for j in test_1_pix:
    img = load_image(j)
    result = model.predict(img)
    if(result[0]==0):
        p1=p1+1
p2=0
for j in test_2_pix:
    img = load_image(j)
    result = model.predict(img)
    if(result[0]==1):
        p2=p2+1
```


```python
print(p1)
```

    0
    


```python
print(p2)
```

    7
    
