import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, RepeatVector, TimeDistributed,
    BatchNormalization, Dropout
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def create_autoencoder_model(input_shape, encoding_dim=64):
    """
    Create an LSTM-based Autoencoder model
    
    Parameters:
    input_shape: Shape of input data (time_steps, features)
    encoding_dim: Dimension of the encoded representation
    """
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Encoder
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(64, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Encoded representation
    encoded = Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = RepeatVector(input_shape[0])(encoded)
    
    x = LSTM(64, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    decoded = TimeDistributed(Dense(input_shape[1]))(x)
    
    # Create autoencoder model
    autoencoder = Model(inputs, decoded)
    
    # Create encoder model
    encoder = Model(inputs, encoded)
    
    return autoencoder, encoder

def create_classifier_from_encoder(encoder, num_classes=10):
    """
    Create a classifier using the encoder's output
    
    Parameters:
    encoder: Trained encoder model
    num_classes: Number of output classes (genres)
    """
    
    # Get the encoder's output
    encoded_input = Input(shape=(encoder.output_shape[1],))
    
    # Classification layers
    x = Dense(256, activation='relu')(encoded_input)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create classifier model
    classifier = Model(encoded_input, outputs)
    
    return classifier

def train_autoencoder(X_train, X_val, model_save_path='model_autoencoder.h5'):
    """
    Train the autoencoder model
    
    Parameters:
    X_train: Training data
    X_val: Validation data
    model_save_path: Path to save the trained model
    """
    
    # Create model
    autoencoder, encoder = create_autoencoder_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Compile model
    autoencoder.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse'
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = autoencoder.fit(
        X_train, X_train,  # Autoencoder learns to reconstruct input
        validation_data=(X_val, X_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return autoencoder, encoder, history

def train_classifier(encoder, X_train, y_train, X_val, y_val, model_save_path='model_classifier.h5'):
    """
    Train the classifier using the encoder's output
    
    Parameters:
    encoder: Trained encoder model
    X_train: Training data
    y_train: Training labels
    X_val: Validation data
    y_val: Validation labels
    model_save_path: Path to save the trained model
    """
    
    # Create classifier
    classifier = create_classifier_from_encoder(encoder)
    
    # Compile model
    classifier.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Get encoded representations
    X_train_encoded = encoder.predict(X_train)
    X_val_encoded = encoder.predict(X_val)
    
    # Train classifier
    history = classifier.fit(
        X_train_encoded, y_train,
        validation_data=(X_val_encoded, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return classifier, history

def evaluate_autoencoder_classifier(encoder, classifier, X_test, y_test):
    """
    Evaluate the autoencoder and classifier
    
    Parameters:
    encoder: Trained encoder model
    classifier: Trained classifier model
    X_test: Test data
    y_test: Test labels
    """
    
    # Get encoded representations
    X_test_encoded = encoder.predict(X_test)
    
    # Evaluate classifier
    test_loss, test_acc = classifier.evaluate(X_test_encoded, y_test, verbose=1)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Get predictions
    y_pred = classifier.predict(X_test_encoded)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return y_pred_classes, y_pred 