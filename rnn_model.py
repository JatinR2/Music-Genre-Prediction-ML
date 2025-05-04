import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Bidirectional, 
    Input, BatchNormalization, Attention,
    Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

def create_rnn_model(input_shape, num_classes=10):
    """
    Create a Bidirectional LSTM model with attention mechanism
    
    Parameters:
    input_shape: Shape of input data (time_steps, features)
    num_classes: Number of output classes (genres)
    """
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First Bidirectional LSTM layer
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Attention mechanism
    attention = Dense(1, activation='tanh')(x)
    attention = tf.keras.layers.Flatten()(attention)
    attention_weights = tf.keras.layers.Activation('softmax')(attention)
    attention_weights = tf.keras.layers.RepeatVector(128)(attention_weights)
    attention_weights = tf.keras.layers.Permute([2, 1])(attention_weights)
    
    # Apply attention weights
    x = tf.keras.layers.Multiply()([x, attention_weights])
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_rnn_model(X_train, y_train, X_val, y_val, model_save_path='model_rnn.h5'):
    """
    Train the RNN model
    
    Parameters:
    X_train: Training data
    y_train: Training labels
    X_val: Validation data
    y_val: Validation labels
    model_save_path: Path to save the trained model
    """
    
    # Create model
    model = create_rnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Compile model
    model.compile(
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
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

def evaluate_rnn_model(model, X_test, y_test):
    """
    Evaluate the trained RNN model
    
    Parameters:
    model: Trained model
    X_test: Test data
    y_test: Test labels
    """
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    return y_pred_classes, y_pred

if __name__ == "__main__":
    # Example usage
    # Load your data here
    # X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Train model
    # model, history = train_rnn_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    # y_pred_classes, y_pred = evaluate_rnn_model(model, X_test, y_test)
    pass 