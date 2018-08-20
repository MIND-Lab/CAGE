'''
Created on Nov, 2016

@author: hugo

'''




from __future__ import absolute_import
from keras import backend as KBack
from keras.layers import Input, Dense, merge, Lambda, add
#from heraspy.callback import HeraCallback
from keras.models import Model
from keras.optimizers import Adadelta, Adam, Adagrad, RMSprop, SGD
from keras.models import load_model as load_keras_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import callbacks

import numpy as np
import copy

from ..utils.keras_utils import Dense_tied, KCompetitive, contractive_loss, CustomModelCheckpoint

#%%


def fit_quadruple_hyperas( input_size, n_dim, comp_topk=None, ctype=None, kfactor = 6.26, alpha = 1, save_model='best_model', gamma = 1.,
                          num_nodes = 1, num_edges = 1, train_data = None, test_data = None, val_split = 0.0 , nb_epoch=50, batch_size=100,
                          contractive=None, optimizer = None, lr = None,
                          select_diff = 0, select_loss = 10, select_graph_np_diff = 2, select_loss_gamma = 0, select_diff_gamma = 0):

        
        input_layer = Input(shape=(input_size,))

        if ctype == None:
            act = 'sigmoid'
        elif ctype == 'kcomp':
            act = 'tanh'
        elif ctype == 'ksparse':
            act = 'linear'
        else:
            raise Exception('unknown ctype: %s' % ctype)
        encoded_layer = Dense(n_dim, activation=act, kernel_initializer="glorot_normal", name="Encoded_Layer")
        encoded = encoded_layer(input_layer)

        if comp_topk:
            print ('add k-competitive layer')
            encoded = KCompetitive(comp_topk, ctype, kfactor)(encoded)

        decoded = Dense_tied(input_size, activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        autoencoder = Model(outputs=decoded, inputs=input_layer)

        encoder = Model(outputs=encoded, inputs=input_layer)

        encoded_input = Input(shape=(n_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)

#%%
            
        def loss_structure_1(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
            y_pred: Contains ys2-ys1
            y_true: Contains (yt2-yt1) difference struct vector
            '''
            min_batch_size = KBack.shape(y_true)[0]   
            return KBack.sqrt(KBack.square(KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1),[min_batch_size, 1]) - y_true))
        
        def loss_structure_2(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
            y_pred: Contains ys2-ys1
            y_true: Contains (yt2-yt1) difference struct vector
            '''
            min_batch_size = KBack.shape(y_true)[0]   
            return KBack.sqrt(KBack.square(KBack.reshape(KBack.sum((y_pred), axis=-1),[min_batch_size, 1]) - y_true))
        
        def simple_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.reshape(y_pred,[min_batch_size, 1]) - y_true)
        
        def squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true))
        
        def mean_squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.mean(KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true)))
        
        def sqrt_mean_squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.mean(KBack.sqrt(KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true))))
        
        def sum_squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.sum(KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true)))
        
        def sqrt_sum_squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.sum(KBack.sqrt(KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true))))
        
        def sqrt_squared_loss(y_true, y_pred):
            min_batch_size = KBack.shape(y_true)[0]   
            return (KBack.sqrt(KBack.square(KBack.reshape(y_pred,[min_batch_size, 1]) - y_true)))
        
        option_loss = {
           0 : simple_loss,
           1 : squared_loss,
           2 : mean_squared_loss,
           3 : sqrt_mean_squared_loss,
           4 : sum_squared_loss,
           5 : sqrt_sum_squared_loss,
           6 : sqrt_squared_loss,
           7 : loss_structure_1,
           8 : loss_structure_2
        }
        
        
        x = Input(shape=(encoder.layers[0].input_shape[1],))
        y = encoder(x)
        x_hat = decoder(y)
        autoencoder_quadruple = Model(inputs=x, outputs=[x_hat, y])
        
        
        x_in_1 = Input(shape=(input_size,), name='x_in_1')
        x_in_2 = Input(shape=(input_size,), name='x_in_2')
        # Process inputs
        [x_hat1, y1] = autoencoder_quadruple(x_in_1)
        [x_hat2, y2] = autoencoder_quadruple(x_in_2)
        
 #%%       
        def simple_difference(input_tuple):
            # return vector
            a,b = input_tuple
            return a-b
        
        def squared_difference(input_tuple):
            # return vector
            a,b = input_tuple
            return KBack.square(a-b)
        
        def mean_squared_difference(input_tuple):
            # return number
            a,b = input_tuple
            return KBack.mean(KBack.square(a-b), axis=-1)
        
        def sqrt_mean_squared_difference(input_tuple):
            # return number
            a,b = input_tuple
            return KBack.mean(KBack.sqrt(np.square(a-b)), axis=-1)
        
        def sum_squared_difference(input_tuple):
            # return number
            a,b = input_tuple
            return KBack.sum(KBack.square(a-b), axis=-1)
        
        def sqrt_sum_squared_difference(input_tuple):
            # return number
            a,b = input_tuple
            return KBack.sum(KBack.sqrt(KBack.square(a-b)), axis=-1)
        
        def sqrt_squared_difference(input_tuple):
            # return vector
            a,b = input_tuple
            return KBack.sqrt(KBack.square(a-b))
        
        option_diff = {
           0 : mean_squared_difference,
           1 : sqrt_mean_squared_difference,
           2 : sum_squared_difference,
           3 : sqrt_sum_squared_difference
        }
        
#        minus_y1 = Lambda(lambda x: -x)(y1)
#        text_diff = add([y2,minus_y1])
        
        text_diff = merge([y2, y1], mode=option_diff[select_diff], output_shape=lambda L: L[1])
        y_diff = merge([y2, y1], mode=option_diff[select_diff_gamma], output_shape=lambda L: L[1])
        model = Model(inputs=[x_in_1,x_in_2], outputs=[x_hat1,x_hat2,text_diff, y_diff])
        
        
        def np_mean_squared_difference(a,b):
            return np.mean(np.square(a-b), axis=-1)
        
        def np_sqrt_mean_squared_difference(a,b):
            return np.mean(np.sqrt(np.square(a-b)), axis=-1)
        
        def np_sum_squared_difference(a,b):
            return np.sum(np.square(a-b), axis=-1)
        
        def np_sqrt_sum_squared_difference(a,b):
            return np.sum(np.sqrt(np.square(a-b)), axis=-1)
        
        option_graph_np_diff = {
           0 : np_mean_squared_difference,
           1 : np_sqrt_mean_squared_difference,
           2 : np_sum_squared_difference,
           3 : np_sqrt_sum_squared_difference
        }
        
        
        
        train_data_temp = copy.copy(train_data)
        
        train_input = train_data[0]
        train_reconstructed = train_data[1]
        graph_vector_input = train_data_temp[2]
        diff_graph_embedding = option_graph_np_diff[select_graph_np_diff](graph_vector_input[0],graph_vector_input[1])
        train_output=train_reconstructed+[diff_graph_embedding]+[np.zeros(len(diff_graph_embedding))]
        
        
        if (optimizer == None or optimizer.lower() == "adadelta"):
            if (lr is None):
                optimizer_model = Adadelta()
            else:
                optimizer_model = Adadelta(lr = lr)
        elif (optimizer.lower() == "adam"):
            if (lr is None):
                optimizer_model = Adam()
            else:
                optimizer_model = Adam(lr = lr)
        elif (optimizer.lower() == "adagrad"):
            if (lr is None):
                optimizer_model = Adagrad()
            else:
                optimizer_model = Adagrad(lr = lr)
        elif (optimizer.lower() == "rmsprop"):
            if (lr is None):
                optimizer_model = RMSprop()
            else:
                optimizer_model = RMSprop(lr = lr)
        elif (optimizer.lower() == "sgd"):
            if (lr is None):
                optimizer_model = SGD(decay=1e-5, momentum=0.99, nesterov=True)
            else:
                optimizer_model = SGD(lr = lr,decay=1e-5, momentum=0.99, nesterov=True)
                
#%%
                
        if contractive:
            print ('Using contractive loss, lambda: %s' % contractive)
            model.compile(optimizer=optimizer_model, loss=contractive_loss( contractive))
        else:
            print ('Using binary crossentropy')
            model.compile(optimizer=optimizer_model, loss=['binary_crossentropy','binary_crossentropy',option_loss[select_loss],option_loss[select_loss_gamma]], loss_weights=[1., 1., alpha, gamma]) # kld, binary_crossentropy, mse

        history = model.fit(train_input,  train_output,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split = val_split,
                        callbacks=[
                                    callbacks.TerminateOnNaN(),
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3*2, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5*2, verbose=1, mode='auto')#,
                                    #CustomModelCheckpoint(encoder, 'models/dcc/model', monitor='val_loss', save_best_only=True, mode='auto')
                        
                                    ]
                        )
        _Y = encoder.predict(test_data)
        
        #%%
    
        return history, _Y, model

#%%

class AutoEncoder(object):
    """AutoEncoder for topic modeling.

        Parameters
        ----------
        """

    def __init__(self, input_size, dim, comp_topk=None, ctype=None, kfactor = 6.26, alpha = 1, save_model='best_model', num_nodes = 1, num_edges = 1):
        self.input_size = input_size
        self.dim = dim
        self.comp_topk = comp_topk
        self.ctype = ctype
        self.save_model = save_model
        self.kfactor = kfactor
        self.alpha = alpha
        self.num_nodes = num_nodes
        self.num_edges = num_edges

#        self.build()
#        self.build_double_ae()
        self._model_quadruple = self.build_quadruple_ae()
        

    def build(self):
        # this is our input placeholder
        input_layer = Input(shape=(self.input_size,))

        # "encoded" is the encoded representation of the input
        if self.ctype == None:
            act = 'sigmoid'
        elif self.ctype == 'kcomp':
            act = 'tanh'
        elif self.ctype == 'ksparse':
            act = 'linear'
        else:
            raise Exception('unknown ctype: %s' % self.ctype)
        encoded_layer = Dense(self.dim, activation=act, kernel_initializer="glorot_normal", name="Encoded_Layer")
        encoded = encoded_layer(input_layer)

        if self.comp_topk:
            print ('add k-competitive layer')
            encoded = KCompetitive(self.comp_topk, self.ctype, self.kfactor)(encoded)

        # "decoded" is the lossy reconstruction of the input
        # add non-negativity contraint to ensure probabilistic interpretations
        decoded = Dense_tied(self.input_size, activation='sigmoid', tied_to=encoded_layer, name='Decoded_Layer')(encoded)

        # this model maps an input to its reconstruction
        self.autoencoder = Model(outputs=decoded, inputs=input_layer)

        # this model maps an input to its encoded representation
        self.encoder = Model(outputs=encoded, inputs=input_layer)

        # create a placeholder for an encoded input
        encoded_input = Input(shape=(self.dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = self.autoencoder.layers[-1]
        # create the decoder model
        self.decoder = Model(outputs=decoder_layer(encoded_input), inputs=encoded_input)
        
        
    def build_double_ae(self):
         # Build the autoencoder with two output
         
         # Input
        x = Input(shape=(self.encoder.layers[0].input_shape[1],))
        # Generate embedding
        y = self.encoder(x)
        # Generate reconstruction
        x_hat = self.decoder(y)
        # Autoencoder Model
        self.autoencoder_double = Model(input=x, output=[x_hat, y])
        
        
        x_in_1 = Input(shape=(self.input_size,), name='x_in_1')
        x_in_2 = Input(shape=(self.input_size,), name='x_in_2')
        # Process inputs
        [x_hat1, y1] = self.autoencoder_double(x_in_1)
        [x_hat2, y2] = self.autoencoder_double(x_in_2)
        
         # Model
        self._model_double = Model(inputs=[x_in_1,x_in_2], outputs=[x_hat1,x_hat2])
#        self._model_double = Model(inputs=[x_in_1], outputs=[x_hat1])
        print(self._model_double.summary())
        
#        return autoencoder_double

    def build_quadruple_ae(self):
         # Build the autoencoder with two output
         
         # Input
        x = Input(shape=(self.encoder.layers[0].input_shape[1],))
        # Generate embedding
        y = self.encoder(x)
        # Generate reconstruction
        x_hat = self.decoder(y)
        # Autoencoder Model
        self.autoencoder_quadruple = Model(input=x, output=[x_hat, y])
        
        
        x_in_1 = Input(shape=(self.input_size,), name='x_in_1')
        x_in_2 = Input(shape=(self.input_size,), name='x_in_2')
        # Process inputs
        [x_hat1, y1] = self.autoencoder_quadruple(x_in_1)
        [x_hat2, y2] = self.autoencoder_quadruple(x_in_2)
        
        # Graph Embedding Input
#        ge_1 = Input(shape=(self.input_size,), name='ge_1')
#        ge_2 = Input(shape=(self.input_size,), name='ge_2')
        
        def difference(input_tuple):
            a,b = input_tuple
            return a-b
        
        text_diff = merge([y2, y1], mode=difference, output_shape=lambda L: L[1])
#        struct_diff = merge([ge_2, ge_1], mode=difference, output_shape=lambda L: L[1])
        
        
         # Model
        self._model_quadruple = Model(inputs=[x_in_1,x_in_2], outputs=[x_hat1,x_hat2,text_diff])
#        self._model_double = Model(inputs=[x_in_1], outputs=[x_hat1])
        print(self._model_quadruple.summary())
        
        return self._model_quadruple
        
        
    
        
        
    def fit_quadruple(self, model, train_data, val_split, nb_epoch=50, batch_size=100, contractive=None, optimizer = None, lr = None):
        
        def loss_structure(y_true, y_pred):
            ''' Hack: This fn doesn't accept additional arguments. We use y_true to pass them.
            y_pred: Contains ys2-ys1
            y_true: Contains (yt2-yt1) difference struct vector
            '''
            min_batch_size = KBack.shape(y_true)[0]   
            return KBack.sqrt(KBack.square(KBack.reshape(KBack.sum(KBack.square(y_pred), axis=-1),[min_batch_size, 1]) - y_true))
#            return KBack.sqrt(KBack.square(KBack.reshape(KBack.sum(KBack.sqrt(KBack.square(y_pred)), axis=-1),[min_batch_size, 1]) - y_true))
    
#        model = self._model_quadruple
        
        train_data_temp = copy.copy(train_data)
        
        
        train_input = train_data[0]
        train_reconstructed = train_data[1]
        graph_vector_input = train_data_temp[2]
        graph_embedding = np.sum(np.square(graph_vector_input[0]-graph_vector_input[1]), axis=-1)
        train_output=train_reconstructed+[graph_embedding]

#        train_output = train_data_temp[1].append(graph_embedding)
        
        
        if (optimizer == None or optimizer.lower() == "adadelta"):
    #            optimizer = Adadelta(lr=2.)
            if (lr is None):
                optimizer = Adadelta()
            else:
                optimizer = Adadelta(lr = lr)
        elif (optimizer.lower() == "adam"):
            if (lr is None):
                optimizer = Adam()
            else:
                optimizer = Adam(lr = lr)
        elif (optimizer.lower() == "adagrad"):
            if (lr is None):
                optimizer = Adagrad()
            else:
                optimizer = Adagrad(lr = lr)
        elif (optimizer.lower() == "rmsprop"):
            if (lr is None):
                optimizer = RMSprop()
            else:
                optimizer = RMSprop(lr = lr)
        if contractive:
            print ('Using contractive loss, lambda: %s' % contractive)
            model.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print ('Using binary crossentropy')
            model.compile(optimizer=optimizer, loss=['binary_crossentropy','binary_crossentropy',loss_structure], loss_weights=[1., 1., self.alpha]) # kld, binary_crossentropy, mse
    
    #        herasCallback = HeraCallback(
    #            'model-key',
    #            'localhost',
    #            4000
    #        )
#        print(train_data_temp[0])
#        print(len(train_input))
#        print(len(train_output))
        history = model.fit(train_input,  train_output,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split = val_split,
                        callbacks=[
    #                                    herasCallback,
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3*2, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5*2, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                        ).history
        
        n_iters =  len(history["loss"])
            
    
        return self,history
    
    
           
    def fit_double(self, train_X, val_split, nb_epoch=50, batch_size=100, contractive=None, optimizer = None, lr = None, double = True):
        if (double):
            model = self._model_double
        else:
            model = self.autoencoder
        
        
        if (optimizer == None or optimizer.lower() == "adadelta"):
    #            optimizer = Adadelta(lr=2.)
            optimizer = Adadelta(lr = lr)
        elif (optimizer.lower() == "adam"):
            optimizer = Adam(lr = lr)
        elif (optimizer.lower() == "adagrad"):
            optimizer = Adagrad(lr = lr)
        if contractive:
            print ('Using contractive loss, lambda: %s' % contractive)
            model.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print ('Using binary crossentropy')
            model.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse
    
    #        herasCallback = HeraCallback(
    #            'model-key',
    #            'localhost',
    #            4000
    #        )
    
        history = model.fit(train_X[0], train_X[1],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split = val_split,
                        callbacks=[
    #                                    herasCallback,
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                        ).history
        
        n_iters =  len(history["loss"])
            
    
        return self,n_iters

    def fit(self, train_X, val_X, nb_epoch=50, batch_size=100, contractive=None, optimizer = None, lr = None, double = False):
        if (double):
            model = self._model_double
        else:
            model = self.autoencoder
        
        
        if (optimizer == None or optimizer.lower() == "adadelta"):
#            optimizer = Adadelta(lr=2.)
            optimizer = Adadelta(lr = lr)
        elif (optimizer.lower() == "adam"):
            optimizer = Adam(lr = lr)
        elif (optimizer.lower() == "adagrad"):
            optimizer = Adagrad(lr = lr)
        if contractive:
            print ('Using contractive loss, lambda: %s' % contractive)
            model.compile(optimizer=optimizer, loss=contractive_loss(self, contractive))
        else:
            print ('Using binary crossentropy')
            model.compile(optimizer=optimizer, loss='binary_crossentropy') # kld, binary_crossentropy, mse

#        herasCallback = HeraCallback(
#            'model-key',
#            'localhost',
#            4000
#        )

        history = model.fit(train_X[0], train_X[1],
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_data=(val_X[0], val_X[1]),
                        callbacks=[
#                                    herasCallback,
                                    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.01),
                                    EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=5, verbose=1, mode='auto'),
                                    CustomModelCheckpoint(self.encoder, self.save_model, monitor='val_loss', save_best_only=True, mode='auto')
                        ]
                        ).history
        
        n_iters =  len(history["loss"])
        

        return self,n_iters

def save_ae_model(model, model_file):
    model.save(model_file)

def load_ae_model(model_file):
    return load_keras_model(model_file, custom_objects={"KCompetitive": KCompetitive})
