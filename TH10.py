import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import classification_report
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from PIL import Image

# Set seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Paths
train_dir = "./dataset/Train"
val_dir = "./dataset/Validation"

# ===================================================
# Part 1: CNN for Classification
# ===================================================
def train_cnn():
    print("Training CNN Model...")

    # Data preprocessing
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=16, class_mode='binary')
    val_data = val_datagen.flow_from_directory(val_dir, target_size=(150, 150), batch_size=16, class_mode='binary')

    # Build CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train model
    model.fit(train_data, epochs=10, validation_data=val_data)

    # Evaluate model
    val_data.reset()
    predictions = model.predict(val_data)
    y_pred = (predictions > 0.5).astype(int)
    y_true = val_data.classes
    print(classification_report(y_true, y_pred, target_names=val_data.class_indices.keys()))

    # Save model
    model.save("cnn_dog_cat_classifier.keras")
    print("CNN Model saved as cnn_dog_cat_classifier.keras")

    return model


def predict_cnn(model, image_path):
    print(f"Predicting with CNN for {image_path}...")
    if not os.path.exists(image_path):
        print(f"Error: Image at {image_path} does not exist!")
        return

    try:
        img = load_img(image_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        if prediction > 0.5:
            print("CNN Prediction: It's a Dog!")
        else:
            print("CNN Prediction: It's a Cat!")
    except Exception as e:
        print(f"Error while predicting: {e}")


# ===================================================
# Part 2: Faster R-CNN for Object Detection
# ===================================================
def train_faster_rcnn():
    print("Training Faster R-CNN Model...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load pre-trained Faster R-CNN model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 3  # [background, cat, dog]
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)

    # Example: Dummy training loop (requires dataset preparation in COCO/Pascal VOC format)
    optimizer = torch.optim.SGD(params=[p for p in model.parameters() if p.requires_grad],
                                lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Replace this with your DataLoader
    train_loader = []  # Placeholder for your DataLoader

    for epoch in range(10):  # Replace 10 with desired number of epochs
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/10], Loss: {epoch_loss}")

    torch.save(model.state_dict(), "faster_rcnn_dog_cat.pth")
    print("Faster R-CNN Model saved as faster_rcnn_dog_cat.pth")

    return model


def predict_faster_rcnn(model, image_path):
    print(f"Predicting with Faster R-CNN for {image_path}...")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()

    try:
        img = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img).to(device).unsqueeze(0)

        with torch.no_grad():
            predictions = model(img_tensor)
            print(predictions)
    except Exception as e:
        print(f"Error while predicting: {e}")


# ===================================================
# Main Execution
# ===================================================
if __name__ == "__main__":
    # Train and test CNN
    cnn_model = train_cnn()
    predict_cnn(cnn_model, "./dataset/Validation/Dogs/dog.4007.jpg")

    # Train and test Faster R-CNN
    # Note: You must prepare DataLoader for Faster R-CNN training.
    # Dummy example provided here.
    faster_rcnn_model = train_faster_rcnn()
    predict_faster_rcnn(faster_rcnn_model, "./dataset/Validation/Cats/cat.4017.jpg")
