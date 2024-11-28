# %%
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input, Flatten, Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
from PIL import Image, ImageDraw
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# %%
def load_annotations_from_excel(excel_path):
  annotations_df = pd.read_csv(excel_path)
  return annotations_df
# %%
def load_image(image_path, target_size=(128,128)):
  image = Image.open(image_path).convert('RGB')
  image = image.resize(target_size)
  image = np.array(image) / 255.0
  return image
# %%
def dataset_generator_with_coordinates(image_dir, annotations_df, batch_size, target_size =(128,128), augment = False):
  images_filenames = annotations_df['filename'].values
  annotations=annotations_df[['xmin', 'xmax' , 'ymin' , 'ymax']].values

  datagen = None
  if augment:
    datagen = ImageDataGenerator(
      rotation_range = 20,
      width_shift_range = 0.2,
      height_shift_range = 0.2,
      shear_range = 0.2,
      zoom_range = 0.2,
      horizontal_flip = True,
      fill_mode = 'nearest'
    )
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
          if augment and datagen is not None:
            augmented = next(datagen.flow(np.expand_dims(image, axis=0),batch_size = 1))
            image = augmented[0]
          images.append(image)
          coords.append([
                        annotation[0] / target_size[0],  # xmin normalized
                        annotation[1] / target_size[0],  # xmax normalized
                        annotation[2] / target_size[1],  # ymin normalized
                        annotation[3] / target_size[1],  # ymax normalized
                    ])
        except (FileNotFoundError, OSError):
          continue
      yield np.array(images), np.array(coords)

# %%
def lightweight_unet(input_shape=(128, 128, 3)):
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    inputs = Input(shape=input_shape)
    
    # Encoder (Simple CNN)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(inputs)
    c1 = MaxPooling2D((2, 2))(c1)  # Output: (64, 64, 16)
    c1 = Dropout(0.3)(c1)
    
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(c1)
    c2 = MaxPooling2D((2, 2))(c2)  # Output: (32, 32, 32)
    c2 = Dropout(0.3)(c2)
    
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(c2)
    c3 = MaxPooling2D((2, 2))(c3)  # Output: (16, 16, 64)
    c3 = Dropout(0.3)(c3)

    # Bottleneck
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(c3)  # Output: (16, 16, 128)
    c4 = Dropout(0.3)(c4)

    # Decoder
    u3 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)  # Upsample to (32, 32, 64)
    u3 = concatenate([u3, c2])  # Concatenate with encoder layer c2
    u3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(u3)
    u3 = Dropout(0.3)(u3)
    
    u2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(u3)  # Upsample to (64, 64, 32)
    u2 = concatenate([u2, c1])  # Concatenate with encoder layer c1
    u2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(u2)
    u2 = Dropout(0.3)(u2)

    u1 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(u2)  # Upsample to (128, 128, 16)
    u1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-2))(u1)
    u1 = Dropout(0.3)(u1)

    pooled = GlobalAveragePooling2D()(u1)

    dense1 = Dense(64, activation = 'relu')(pooled)
    dense2 = Dense(32, activation = 'relu')(dense1)
    
    # Output layer for bounding box coordinates
    output = Dense(4, activation='linear')(dense2)  # Output 4 values (xmin, ymin, xmax, ymax)

    model = Model(inputs, output)
    return model

# %%
def train_lightweight_model(train_image_dir, train_excel_path, val_image_dir, val_excel_path, batch_size=8, epochs=10):
  from tensorflow.keras.losses import Huber
  input_shape = (128,128,3)
  model = lightweight_unet(input_shape)
  model.compile(optimizer=Adam(learning_rate=0.0001), loss=Huber(), metrics=['mae'])

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
# %%
train_image_dir = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\train'
train_excel_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\Annotation csv\train_annotations.csv'
val_image_dir = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\valid'
val_excel_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\Annotation csv\valid_annotations.csv'
# %%
model,history = train_lightweight_model(train_image_dir, train_excel_path, val_image_dir, val_excel_path, batch_size=8, epochs=10)
# %%
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# %%
def calculate_iou(box1 , box2):
  x1 = max(box1[0],box2[0])
  y1 = max(box1[1],box2[1])
  x2 = min(box1[2],box2[2])
  y2 = min(box1[3],box2[3])

  intersection = max(0 , x2-x1)*max(0 , y2-y1)

  box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
  box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])

  union = box1_area + box2_area - intersection

  iou = intersection /union if union > 0 else 0
  return iou
# %%
def evaluate_model(test_image_dir , test_excel_path, model , target_size=(128,128), iou_threshold = 0.5):
  test_annotations_df = load_annotations_from_excel(test_excel_path)

  correct_predictions = 0
  total_predictions = len(test_annotations_df)
  print(f"Total Predictions {total_predictions}")

  for index, row in test_annotations_df.iterrows():
    image_filename = row['filename']
    ground_truth_box = [
            row['xmin'] / row['width'],
            row['ymin'] / row['height'],
            row['xmax'] / row['width'],
            row['ymax'] / row['height']
        ]

    image_path = os.path.join(test_image_dir , image_filename)
    original_image = Image.open(image_path).convert('RGB')
    original_size = original_image.size

    image_resized = original_image.resize(target_size)
    image_array = np.array(image_resized, dtype = np.float64)/255.0
    image_array = np.expand_dims(image_array , axis =0)

    predicted_box= model.predict(image_array)[0]

    x_scale= original_size[0]/target_size[0]
    y_scale= original_size[1]/target_size[1]

    predicted_box_rescaled = [
      (predicted_box[0]*original_size[0]),
      (predicted_box[1]*original_size[1]),
      (predicted_box[2]*original_size[0]),
      (predicted_box[3]*original_size[1])]
    
    ground_truth_box_rescaled = [
            ground_truth_box[0] * original_size[0],
            ground_truth_box[1] * original_size[1],
            ground_truth_box[2] * original_size[0],
            ground_truth_box[3] * original_size[1]
        ]
    iou = calculate_iou(predicted_box_rescaled, ground_truth_box_rescaled)
    if iou <= iou_threshold:
      correct_predictions = correct_predictions+1
  
  accuracy= (correct_predictions / total_predictions) *100
  print(f"Model Accuracy: {accuracy:.2f}%")
  return accuracy
# %%
test_image_dir = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\test'
test_excel_path = r'C:\Users\HP\Desktop\My Documents\acharya shit - Copy\Major Project\DXS PROJ FILE\Annotation csv\test_annotations.csv'
model_accuracy = evaluate_model(test_image_dir , test_excel_path , model , target_size=[128,128], iou_threshold=0.5)
# %%
