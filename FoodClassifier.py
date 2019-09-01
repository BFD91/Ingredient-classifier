import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from PIL import Image
import numpy as np
from skimage import transform
import pandas as pd

num_classes = input('Enter number of classes to be classified: ')

train_dir = 'Foods train'
validation_dir = 'Foods validation'
#test_dir = 'Foods test'

def load_image(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (100, 100, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

model = tf.keras.Sequential()

model.add(layers.Conv2D(32, (3, 3), padding='same',
                 input_shape=(100,100,3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy', 
              metrics = ['acc'])
              

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 45,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
                                   
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 32,
                                                    class_mode = 'categorical', 
                                                    target_size = (100, 100))
                                                    
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                    batch_size = 8,
                                                    class_mode = 'categorical', 
                                                    target_size = (100, 100))
                                                    
#test_generator = test_datagen.flow_from_directory(test_dir,
#                                                    batch_size = 8,
#                                                    class_mode = 'categorical', 
#                                                    target_size = (100, 100))
                                                    
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    verbose = 2,
                    epochs=10
)

#test_generator.reset()
#pred=model.predict_generator(test_generator,
#steps=STEP_SIZE_TEST,
#verbose=1)

#predicted_class_indices=np.argmax(pred,axis=1)

#labels = (train_generator.class_indices)
#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]

#filenames=test_generator.filenames
#print(filenames)
#print(predictions)
#print(len(filenames))
#print(len(predictions))
#results=pd.DataFrame({"Filename":filenames,
#                      "Predictions":predictions})
#results.to_csv("predictions.csv",index=False)
model.save('Food_classifier.h5')