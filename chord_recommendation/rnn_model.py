from typing import *
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from chord_recommendation.configs import *
from chord_recommendation.utils import *

class RnnModel():
    def construct(self, dropout_rate: float):
        ''' Construct the model.
        - dropout_rate: Dropout rate.
        '''
        model_input = layers.Input(batch_shape=(None, N_STEPS, N_INPUT))
        x = layers.SimpleRNN(
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

    def predict(self, chords: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], np.ndarray]:
        '''
        Arg:
        - chords: A chord sequence/progression
        Returns:
        - sorted_chords: A chord sequence of all possible chords for the next
          step, sorted by the probabilities
        - probab: probability of each chord in sorted order
        '''
        onehot_mat = chords_to_onehot_mat(chords[-N_STEPS:])
        probab = self.predict_onehot_batch(np.array([onehot_mat]))
        probab = probab[0] # fetch the first one, since the batch size is 1
        ind_array = np.argsort(probab)[::-1]
        probab = probab[ind_array]
        sorted_chords = []
        for i in range(len(ind_array)):
            sorted_chords.append(id_to_chord(ind_array[i]))
        return sorted_chords, probab

    def predict_onehot_batch(self, onehot_mats: np.ndarray) -> np.ndarray:
        '''
        Arg:
        - onehot_mats: An array with dimensions:
          [batch_size, n_steps, onehot_len]
        Return:
        - probab: The probabilities of each chord
        '''
        batch_size = onehot_mats.shape[0]
        prediction = self.model.predict(
            onehot_mats,
            batch_size=batch_size,
            verbose=0
        )
        # Fetch all the outputs, but only the last step
        # [[0.1, 0.2, 0.4, ...],
        #  [0.3, 0.1, 0.2, ...]]
        probab = prediction[:, -1]
        return probab
