import tensorflow as tf

NUM_RES_BLOCKS = 4

#Neural network described in the dissertation

class Connect4Zero:    

    def build_convolutional_block(self, inputs):
        x = tf.keras.layers.Conv2D(32, (3, 3), input_shape=(6,7,3,), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
        
    def build_residual_block(self, inputs):
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        #Skip connection
        x = tf.keras.layers.Add()([inputs, x])
        x = tf.keras.layers.Activation('relu')(x)
        
        return x
        
    def build_policy_head(self, inputs):
        x = tf.keras.layers.Conv2D(2, (1, 1), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(7)(x)
        x = tf.keras.layers.Activation('softmax', name="policy_output")(x)
        
        return x
    
    def build_value_head(self, inputs):
        x = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Activation('relu')(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1)(x)
        x = tf.keras.layers.Activation('tanh', name="value_output")(x)
        
        return x
    
    def build(self):
        input_shape = (6, 7, 3,)
        inputs = tf.keras.layers.Input(shape = input_shape)
        
        current_model = self.build_convolutional_block(inputs)
        for _ in range(0, NUM_RES_BLOCKS):
            current_model = self.build_residual_block(current_model)
            
        policy_head = self.build_policy_head(current_model)
        value_head = self.build_value_head(current_model)
        
        model = tf.keras.Model(inputs=inputs,
			outputs=[policy_head, value_head], name="connect4nn")
        
        losses = {
            "policy_output": "categorical_crossentropy",
            "value_output": "mean_squared_error",
        }
        lossWeights = {"policy_output": 1.0, "value_output": 1.0}
        
        model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.0001), 
                      loss=losses, loss_weights=lossWeights)
        
        return model
    
    def load_model(self, path):
        model = self.build()
        model.load_weights(path)
        
        return model
    
    def save_model(self, model, path):
        model.save_weights(path)
