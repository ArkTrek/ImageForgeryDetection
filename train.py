from tensorflow.keras.preprocessing.image import ImageDataGenerator
TRAINING_DIR = 'Image_Dataset/Train'
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,target_size=(100,100),batch_size=5,class_mode='categorical')
VALIDATION_DIR = 'Image_Dataset/Test'
validation_datagen = ImageDataGenerator(rescale=1.0/255.0)
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,target_size=(100,100),batch_size=10,class_mode='categorical')

history = model.fit(
      train_generator,
      steps_per_epoch=100,  # 2000 images = batch_size * steps
      epochs=10,
      validation_data=validation_generator,
      validation_steps=10,
      callbacks=[lr_sched])