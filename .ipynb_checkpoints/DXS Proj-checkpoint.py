import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from PIL import Image, ImageDraw
import os

def load_annotations_from_excel(excel_path):
  annotations_df = pd.read_csv(excel_path)
  return annotations_df


def load_image(image_path, target_size=(128,128)):
  image = Image.open(image_path).convert('RGB')
  image = image.resize(target_size)
  image = np.array(image) / 255.0
  return image


def dataset_generator_with_coordinates(image_dir, annotations_df, batch_size, target_size =(128,128)):
  images_filenames = annotations_df['filename'].values
  annotations=annotations_df[['xmin', 'xmax' , 'ymin' , 'ymax']].values

  while True:
    indices = np.arange(len(images_filenames))
    np.random.shuffle(indices)

    for start_idx in range(0, len(images_filenames), batch_size):
      images = []
      coords = []
      batch_indices = indices[start_idx: start_idx + batch_size]
      for i in batch_indices:
        image_path = os.path.join(image_dir, images_filenames[i])
        annotation = annotations[i]
        try:
          image = load_image(image_path, target_size = target_size)
          images.append(image)
          coords.append(annotation)
        except (FileNotFoundError, OSError):
          continue
      yield np.array(images), np.array(coords)


def lightweight_unet(input_shape=(128, 128, 3)):
    inputs = Input(shape=input_shape)
    
    # Encoder (Simple CNN)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    c1 = MaxPooling2D((2, 2))(c1)  # Output: (64, 64, 16)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    c2 = MaxPooling2D((2, 2))(c2)  # Output: (32, 32, 32)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    c3 = MaxPooling2D((2, 2))(c3)  # Output: (16, 16, 64)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)  # Output: (16, 16, 128)
    
    # Decoder
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)  # Upsample to (32, 32, 64)
    u3 = concatenate([u3, c2])  # Concatenate with encoder layer c2
    u3 = Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    
    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u3)  # Upsample to (64, 64, 32)
    u2 = concatenate([u2, c1])  # Concatenate with encoder layer c1
    u2 = Conv2D(32, (3, 3), activation='relu', padding='same')(u2)

    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u2)  # Upsample to (128, 128, 16)
    u1 = Conv2D(16, (3, 3), activation='relu', padding='same')(u1)
    
    # Output layer for bounding box coordinates
    output = Flatten()(u1)
    output = Dense(4, activation='linear')(output)  # Output 4 values (xmin, ymin, xmax, ymax)

    model = Model(inputs, output)
    return model


def train_lightweight_model(train_image_dir, train_excel_path, val_image_dir, val_excel_path, batch_size=8, epochs=10):
  input_shape = (128,128,3)
  model = lightweight_unet(input_shape)
  model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

  train_annotations_df = load_annotations_from_excel(train_excel_path)
  val_annotations_df = load_annotations_from_excel(val_excel_path)

  train_gen = dataset_generator_with_coordinates(train_image_dir, train_annotations_df, batch_size, target_size=(128,128))
  val_gen = dataset_generator_with_coordinates(val_image_dir, val_annotations_df, batch_size,target_size=(128,128))

  history = model.fit(train_gen,
      validation_data = val_gen,
      validation_steps = len(val_annotations_df) // batch_size,
      epochs = epochs,
      steps_per_epoch = len(train_annotations_df) // batch_size)
  return model,history


def highlight_prediction(image_path , model, target_size=(128,128)):
  image = Image.open(image_path).convert('RGB')
  original_size = image.size
  image_resized = image.resize(target_size)
  image_array = np.array(image) / 255.0
  image_array = np.expand_dims(image, axis=0)

  bbox_coords = model.predict(image_array)[0]

  x_scale = original_size[0] / target_size[0]
  y_scale = original_size[1] / target_size[1]
  xmin = int(bbox_coords[0] * x_scale)
  xmax = int(bbox_coords[1] * x_scale)
  ymin = int(bbox_coords[2] * y_scale)
  ymax = int(bbox_coords[3] * y_scale)

  draw = ImageDraw.Draw(image)
  draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3) #red box with width 3
  return image

train_image_dir = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\archive (4)\train'
train_excel_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\archive (4)\Annotation csv\train_annotations.csv'
val_image_dir = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\archive (4)\valid'
val_excel_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\archive (4)\Annotation csv\valid_annotations.csv'

model,history = train_lightweight_model(train_image_dir, train_excel_path, val_image_dir, val_excel_path, batch_size=8, epochs=10)
test_image_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\archive (4)\test\0009_jpg.rf.957a0a16c5101765b0679e95eb9619a3.jpg'
highlighted_image = highlight_prediction(test_image_path,model)
