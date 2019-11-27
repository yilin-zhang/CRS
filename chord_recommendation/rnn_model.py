from tensorflow import keras
from tensorflow.keras import layers
from chord_recommendation.configs import *
from chord_recommendation.utils import gen_batch

class RnnModel():
    def construct(self, dropout_rate: float):
        ''' Construct the model.
        - dropout_rate: Dropout rate.
        '''
        model_input = layers.Input(batch_shape=(None, N_STEPS, N_INPUT))
        x = layers.LSTM(
            N_NEURONS, activation='elu', return_sequences=True)(model_input)
        output = layers.Dense(
            N_INPUT, activation='softmax', name='output')(x)
        self.model = keras.Model(inputs=model_input, outputs=output)

    def load(self, model_path: str, dropout_rate: float):
        ''' Load the pre-trained model. (You can change the dropout rate here.)
        - model_path: The pre-trained model's path.
        - dropout_rate: Dropout rate.
        '''
        self.construct(dropout_rate)
        model = keras.models.load_model(model_path)
        # Copy all the weights to the new model
        for new_layer, layer in zip(self.model.layers[1:], model.layers[1:]):
            new_layer.set_weights(layer.get_weights())

    def compile(self):
        ''' Compile the model.
        '''
        self.model.compile(
            optimizer=keras.optimizers.Adadelta(),
            loss={
                'output': 'categorical_crossentropy'
            },
            metrics=['accuracy'])
    
    def fit(self,
            train_path,
            valid_path,
            batch_size,
            steps_per_epoch,
            validation_steps,
            n_epochs,
            model_path,
            log_path,
            initial_epoch=0):
        ''' Fit the model.
        - trian_path: The training set's path.
        - valid_path: The validation set's path.
        - bath_size: The batch size.
        - steps_per_epoch: The steps you need to go through a whole training
        set.
        - validation_steps: The steps you need to go through a whole validation
        set.
        - n_epochs: How many epochs do you want.
        - initial_epoch: It depends on your pre-trained model.
        '''

        # Set tensorboard callback
        tb_callback = keras.callbacks.TensorBoard(log_dir=log_path, batch_size=batch_size)
        model_save_path = model_path + "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
        mc_callback = keras.callbacks.ModelCheckpoint(
            filepath=model_save_path, monitor='val_loss')

        # Summary the model
        self.model.summary()

        # Fit model
        gen_train = gen_batch(train_path, N_STEPS, batch_size)
        gen_valid = gen_batch(valid_path, N_STEPS, batch_size)
        self.model.fit_generator(
            gen_train,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epochs,
            validation_data=gen_valid,
            validation_steps=validation_steps,
            callbacks=[tb_callback, mc_callback],
            workers=2,
            use_multiprocessing=True,
            initial_epoch=initial_epoch)
