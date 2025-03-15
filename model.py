import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from joblib import load, dump
import os

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Function to load base models for feature extraction
def load_base_model(model_name):
    if model_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'vggnet':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 'xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    else:
        raise ValueError("Unknown model name")
    return Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

# Load saved ensemble model
ensemble_model_path = r'Models/ensemble_model.joblib'
ensemble = load(ensemble_model_path)

# Load feature extraction models
resnet_model = load_base_model('resnet')
vggnet_model = load_base_model('vggnet')
xception_model = load_base_model('xception')

# Prediction function
def predict_pcos(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features from the image
    resnet_feat = resnet_model.predict(img_array).flatten()
    vggnet_feat = vggnet_model.predict(img_array).flatten()
    xception_feat = xception_model.predict(img_array).flatten()

    # Combine extracted features
    combined_feat = np.concatenate([resnet_feat, vggnet_feat, xception_feat])
    combined_feat = np.expand_dims(combined_feat, axis=0)

    # Predict with the ensemble model
    pred = ensemble.predict(combined_feat)
    return "PCOS Detected" if pred[0] == 0 else "No PCOS Detected"

# Training and feature extraction logic
if __name__ == "__main__":
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    # Set up ImageDataGenerator for loading data
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_data = datagen.flow_from_directory(
        r'Dataset/train',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',  # Binary for PCOS or not PCOS
        shuffle=True
    )

    test_data = datagen.flow_from_directory(
        r'Dataset/test',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    # Directory to save features
    feature_dir = r'SML_DL_PROJECT/features'
    os.makedirs(feature_dir, exist_ok=True)

    def save_features(model, generator, model_name):
        features, labels = extract_features(model, generator)
        np.save(os.path.join(feature_dir, f"{model_name}_features.npy"), features)
        np.save(os.path.join(feature_dir, f"{model_name}_labels.npy"), labels)
        return features, labels

    # Extract features using the models
    def extract_features(model, generator):
        features = []
        labels = []
        for batch_images, batch_labels in generator:
            features.append(model.predict(batch_images))
            labels.extend(batch_labels)
            if len(labels) >= generator.samples:  # Exit loop after processing all data
                break
        return np.vstack(features), np.array(labels)

    # Extract features and save them
    print("Extracting features using ResNet50...")
    resnet_features, labels = save_features(resnet_model, train_data, "resnet")
    print("Extracted features successfully")

    print("Extracting features using VGG16...")
    vggnet_features, _ = save_features(vggnet_model, train_data, "vggnet")
    print("Extracted features successfully")

    print("Extracting features using Xception...")
    xception_features, _ = save_features(xception_model, train_data, "xception")
    print("Extracted features successfully")

    # Combine features
    print("Combining features...")
    combined_features = np.concatenate([resnet_features, vggnet_features, xception_features], axis=1)
    print("Combined features successfully")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

    # Initialize classifiers
    svm = SVC(probability=True)
    dt = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    nb = GaussianNB()
    lr = LogisticRegression(max_iter=1000)

    # Train individual classifiers
    print("Training classifiers...")
    svm.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # Combine classifiers in an ensemble with majority voting
    ensemble = VotingClassifier(estimators=[
        ('svm', svm),
        ('dt', dt),
        ('knn', knn),
        ('nb', nb),
        ('lr', lr)
    ], voting='soft')

    # Train the ensemble
    print("Training ensemble...")
    ensemble.fit(X_train, y_train)

    # Save models
    model_dir = r'SML_DL_PROJECT/Models'
    os.makedirs(model_dir, exist_ok=True)
    dump(svm, os.path.join(model_dir, 'svm_model.joblib'))
    dump(dt, os.path.join(model_dir, 'dt_model.joblib'))
    dump(knn, os.path.join(model_dir, 'knn_model.joblib'))
    dump(nb, os.path.join(model_dir, 'nb_model.joblib'))
    dump(lr, os.path.join(model_dir, 'lr_model.joblib'))
    dump(ensemble, os.path.join(model_dir, 'ensemble_model.joblib'))

    # Evaluate ensemble
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Ensemble accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("Training and model saving completed.")
