---
title: "fgdfgd"
categories: 
  - database
last_modified_at: 2020-02-18T15:40:00+09:00
toc: true
---
```python
import os
train_dir = './input/dataset1/train'
validation_dir ='./input/dataset1/test'

train_1_dir = os.path.join(train_dir, '연갈탄')

train_2_dir = os.path.join(train_dir, '유연탄')

validation_1_dir = os.path.join(validation_dir, '연갈탄')

validation_2_dir = os.path.join(validation_dir, '유연탄')
```


```python
train_1_fnames = os.listdir(train_1_dir)
print(train_1_fnames[:10])

train_2_fnames = os.listdir(train_2_dir)
train_2_fnames.sort()
print(train_2_fnames[:10])
```

    ['0.jpg', '1.jpg', '11.jpg', '12.jpg', '14.jpg', '15.jpg', '17.jpg', '18.jpg', '20.jpg', '21.jpg']
    ['10.png', '100.png', '101.png', '102.png', '104.png', '105.png', '107.png', '108.png', '109.png', '110.png']
    


```python
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
nrows = 4
ncols = 4
pic_index = 0
```


```python
# Set up matplotlib fig, and size it to fit 4x4 pics
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

![programming-language-names](/images/2020-01-05-how-programming-laguages-got-their-names/2020-01-05-how-programming-laguages-got-their-names.jpg)

![png](../assets/images/coal_classification_1_files/coal_classification_1_3_0.png)
    



```python
from tensorflow.keras import layers
from tensorflow.keras import Model
```


```python
# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
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


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        train_dir,  
        target_size=(150, 150),  
        batch_size=35,
        
        class_mode='binary')


validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=35,
        class_mode='binary')
```

    Found 2791 images belonging to 4 classes.
    Found 20 images belonging to 2 classes.
    


```python
history=model.fit(train_generator,
epochs = 30,
validation_data = validation_generator
)
```

    Epoch 1/30
     1/80 [..............................] - ETA: 0s - loss: 0.7511 - acc: 0.0286WARNING:tensorflow:Callbacks method `on_train_batch_end` is slow compared to the batch time (batch time: 0.0080s vs `on_train_batch_end` time: 0.0150s). Check your callbacks.
    80/80 [==============================] - 34s 428ms/step - loss: -37.9602 - acc: 0.7997 - val_loss: 5111.4150 - val_acc: 0.7000
    Epoch 2/30
    80/80 [==============================] - 20s 256ms/step - loss: -47152.6484 - acc: 0.8269 - val_loss: 80258.7188 - val_acc: 0.8500
    Epoch 3/30
    80/80 [==============================] - 20s 252ms/step - loss: -1706032.3750 - acc: 0.8302 - val_loss: 2759146.0000 - val_acc: 0.7500
    Epoch 4/30
    80/80 [==============================] - 21s 262ms/step - loss: -13672292.0000 - acc: 0.8434 - val_loss: 12515001.0000 - val_acc: 0.8000
    Epoch 5/30
    80/80 [==============================] - 21s 261ms/step - loss: -66859588.0000 - acc: 0.8424 - val_loss: 51060016.0000 - val_acc: 0.8000
    Epoch 6/30
    80/80 [==============================] - 21s 259ms/step - loss: -211320944.0000 - acc: 0.8445 - val_loss: 104821992.0000 - val_acc: 0.9000
    Epoch 7/30
    80/80 [==============================] - 20s 255ms/step - loss: -528798208.0000 - acc: 0.8488 - val_loss: 299889856.0000 - val_acc: 0.9000
    Epoch 8/30
    80/80 [==============================] - 21s 258ms/step - loss: -1103508864.0000 - acc: 0.8495 - val_loss: 584753408.0000 - val_acc: 0.8500
    Epoch 9/30
    80/80 [==============================] - 21s 263ms/step - loss: -2099374464.0000 - acc: 0.8538 - val_loss: 3795184896.0000 - val_acc: 0.7000
    Epoch 10/30
    80/80 [==============================] - 22s 279ms/step - loss: -3823550976.0000 - acc: 0.8552 - val_loss: 2309650944.0000 - val_acc: 0.8000
    Epoch 11/30
    80/80 [==============================] - 21s 263ms/step - loss: -6069920768.0000 - acc: 0.8524 - val_loss: 2994722048.0000 - val_acc: 0.8000
    Epoch 12/30
    80/80 [==============================] - 20s 254ms/step - loss: -10054759424.0000 - acc: 0.8552 - val_loss: 6521648640.0000 - val_acc: 0.7500
    Epoch 13/30
    80/80 [==============================] - 20s 251ms/step - loss: -15074971648.0000 - acc: 0.8570 - val_loss: 8365790208.0000 - val_acc: 0.8000
    Epoch 14/30
    80/80 [==============================] - 20s 251ms/step - loss: -21330960384.0000 - acc: 0.8538 - val_loss: 11014299648.0000 - val_acc: 0.8000
    Epoch 15/30
    80/80 [==============================] - 20s 251ms/step - loss: -30139459584.0000 - acc: 0.8563 - val_loss: 24516354048.0000 - val_acc: 0.7500
    Epoch 16/30
    80/80 [==============================] - 20s 252ms/step - loss: -39832506368.0000 - acc: 0.8520 - val_loss: 25376024576.0000 - val_acc: 0.7500
    Epoch 17/30
    80/80 [==============================] - 20s 252ms/step - loss: -54970761216.0000 - acc: 0.8509 - val_loss: 25344641024.0000 - val_acc: 0.8500
    Epoch 18/30
    80/80 [==============================] - 20s 251ms/step - loss: -72534958080.0000 - acc: 0.8520 - val_loss: 39827210240.0000 - val_acc: 0.7500
    Epoch 19/30
    80/80 [==============================] - 20s 248ms/step - loss: -95739789312.0000 - acc: 0.8538 - val_loss: 37891874816.0000 - val_acc: 0.8500
    Epoch 20/30
    80/80 [==============================] - 20s 249ms/step - loss: -122083336192.0000 - acc: 0.8556 - val_loss: 64538791936.0000 - val_acc: 0.7500
    Epoch 21/30
    80/80 [==============================] - 20s 248ms/step - loss: -158390730752.0000 - acc: 0.8545 - val_loss: 82876260352.0000 - val_acc: 0.7500
    Epoch 22/30
    80/80 [==============================] - 20s 247ms/step - loss: -195451158528.0000 - acc: 0.8549 - val_loss: 111006302208.0000 - val_acc: 0.7500
    Epoch 23/30
    80/80 [==============================] - 20s 249ms/step - loss: -235325112320.0000 - acc: 0.8538 - val_loss: 116991787008.0000 - val_acc: 0.7500
    Epoch 24/30
    80/80 [==============================] - 20s 247ms/step - loss: -282936737792.0000 - acc: 0.8520 - val_loss: 135731019776.0000 - val_acc: 0.8000
    Epoch 25/30
    80/80 [==============================] - 20s 247ms/step - loss: -351399608320.0000 - acc: 0.8520 - val_loss: 150021685248.0000 - val_acc: 0.8000
    Epoch 26/30
    80/80 [==============================] - 20s 246ms/step - loss: -425906372608.0000 - acc: 0.8552 - val_loss: 214018408448.0000 - val_acc: 0.8000
    Epoch 27/30
    80/80 [==============================] - 20s 247ms/step - loss: -495676653568.0000 - acc: 0.8567 - val_loss: 212864581632.0000 - val_acc: 0.8500
    Epoch 28/30
    80/80 [==============================] - 20s 246ms/step - loss: -566781476864.0000 - acc: 0.8606 - val_loss: 348776628224.0000 - val_acc: 0.7500
    Epoch 29/30
    80/80 [==============================] - 20s 246ms/step - loss: -676412260352.0000 - acc: 0.8556 - val_loss: 316445720576.0000 - val_acc: 0.8000
    Epoch 30/30
    80/80 [==============================] - 20s 245ms/step - loss: -764879437824.0000 - acc: 0.8549 - val_loss: 433080500224.0000 - val_acc: 0.7500
    


```python
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt


acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) 
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()


plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()


```




    <Figure size 432x288 with 0 Axes>




    
![png](../assets/images/coal_classification_1_files/coal_classification_1_10_1.png)
    



    
![png](../assets/images/coal_classification_1_files/coal_classification_1_10_2.png)
    



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

test_dir ='./input/dataset1/valid'
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

    3
    


```python
print(p2)
```

    6
    


```python

```
