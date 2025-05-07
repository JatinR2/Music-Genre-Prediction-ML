import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, MultiHeadAttention, LayerNormalization,
    Dropout, GlobalAveragePooling1D, Add
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.add1 = Add()
        self.add2 = Add()

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(self.add1([inputs, attn_output]))
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(self.add2([out1, ffn_output]))

def create_transformer_model(input_shape, num_classes=10):
    """
    Create a Transformer model for music genre classification
    
    Parameters:
    input_shape: Shape of input data (time_steps, features)
    num_classes: Number of output classes (genres)
    """
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Transformer blocks
    x = TransformerBlock(
        embed_dim=input_shape[1],  # Use feature dimension as embedding
        num_heads=4,
        ff_dim=256
    )(inputs)
    
    x = TransformerBlock(
        embed_dim=input_shape[1],
        num_heads=4,
        ff_dim=256
    )(x)
    
    # Global average pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_transformer_model(X_train, y_train, X_val, y_val, model_save_path='model_transformer.h5'):
    """
    Train the Transformer model
    
    Parameters:
    X_train: Training data
    y_train: Training labels
    X_val: Validation data
    y_val: Validation labels
    model_save_path: Path to save the trained model
    """
    
    # Create model
    model = create_transformer_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
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

def evaluate_transformer_model(model, X_test, y_test):
    """
    Evaluate the trained Transformer model
    
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