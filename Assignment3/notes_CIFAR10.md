# Result of the Assignment 
Validation Accuracy : 84.2


### Model Architecture :

(Define the model (Achieved Validation ACCURACY of 84.2 in 47th Epoch))

model = Sequential()

model.add(SeparableConv2D(32, 3, 3, border_mode = 'same',  activation = 'relu', kernel_initializer='he_uniform',input_shape=(32, 32, 3))) # Output = 32, RF = 3x3

model.add(BatchNormalization())

model.add(SeparableConv2D(32,3,3, border_mode = 'same', activation= 'relu',kernel_initializer='he_uniform')) #Output = 32, RF = 5x5

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(32,3,3, border_mode = 'same', activation= 'relu',kernel_initializer='he_uniform')) #Output = 32, RF = 7x7

model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2))) #Output = 16, RF = 8x8

model.add(SeparableConv2D(64, 3, 3, border_mode = 'same',  activation= 'relu',kernel_initializer='he_uniform')) #Output = 16, RF = 12x12

model.add(BatchNormalization())

model.add(SeparableConv2D(64,3,3, border_mode = 'same', activation= 'relu', kernel_initializer='he_uniform')) #Output = 16, RF = 16x16

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(64,3,3, border_mode = 'same', activation= 'relu', kernel_initializer='he_uniform')) #Output = 16, RF = 20x20

model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) #Output = 8, RF= 22x22

model.add(SeparableConv2D(128, 3, 3, border_mode = 'same',  activation= 'relu', kernel_initializer='he_uniform')) #Output = 8, RF = 30x30

model.add(BatchNormalization())


model.add(SeparableConv2D(128,3,3, border_mode = 'same', activation= 'relu', kernel_initializer='he_uniform')) #Output = 8, RF = 38x38

model.add(BatchNormalization())

model.add(Dropout(0.1))


model.add(SeparableConv2D(128,3,3, border_mode = 'same', activation= 'relu', kernel_initializer='he_uniform')) #Output = 8, RF = 46x46

model.add(BatchNormalization())


model.add(MaxPooling2D(pool_size=(2, 2))) #Output = 4, RF = 50x50

model.add(Dropout(0.1))

model.add(SeparableConv2D(256,3,3, border_mode='same', activation='relu', kernel_initializer='he_uniform')) #Output = 4, RF = 66x66

model.add(BatchNormalization())


model.add(SeparableConv2D(10,1,1, activation='relu', kernel_initializer='he_uniform')) 

model.add(GlobalAveragePooling2D())

model.add(Activation('softmax'))

(Compile the model)

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])



### Achieved 84.2% accuracy on CIFAR-10 dataset with using 99,653 parameters. I have used Dropout and Augementation to do regularisation which helped in reducing overfitting. Furthermore, I have also used BatchNormalisation to reduce Internal Covariate Shift effect. As a optimiser, I have used ADAM. Output and Receptive Field of each convolution is clearly mentioned against each step in final code which gave validation accuracy of 84.2%. Moreover, there are more chances of improvement in validation accuracy, which I will do in further notebooks. 



#### Logs for model with Convolution layer:
	390/390 [==============================] - 29s 74ms/step - loss: 1.8490 - acc: 0.2902 - val_loss: 1.4411 - val_acc: 0.4641
Epoch 2/50
390/390 [==============================] - 20s 52ms/step - loss: 1.3502 - acc: 0.5119 - val_loss: 1.1353 - val_acc: 0.5843
Epoch 3/50
390/390 [==============================] - 20s 53ms/step - loss: 1.1284 - acc: 0.5983 - val_loss: 0.9388 - val_acc: 0.6665
Epoch 4/50
390/390 [==============================] - 20s 52ms/step - loss: 0.9807 - acc: 0.6568 - val_loss: 0.9093 - val_acc: 0.6805
Epoch 5/50
390/390 [==============================] - 20s 52ms/step - loss: 0.8682 - acc: 0.6993 - val_loss: 0.7664 - val_acc: 0.7347
Epoch 6/50
390/390 [==============================] - 20s 52ms/step - loss: 0.8024 - acc: 0.7251 - val_loss: 0.7255 - val_acc: 0.7503
Epoch 7/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7490 - acc: 0.7435 - val_loss: 0.7032 - val_acc: 0.7592
Epoch 8/50
390/390 [==============================] - 20s 52ms/step - loss: 0.7070 - acc: 0.7584 - val_loss: 0.6817 - val_acc: 0.7665
Epoch 9/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6667 - acc: 0.7721 - val_loss: 0.6712 - val_acc: 0.7718
Epoch 10/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6327 - acc: 0.7824 - val_loss: 0.6263 - val_acc: 0.7872
Epoch 11/50
390/390 [==============================] - 20s 52ms/step - loss: 0.6054 - acc: 0.7933 - val_loss: 0.6389 - val_acc: 0.7877
Epoch 12/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5854 - acc: 0.8000 - val_loss: 0.6163 - val_acc: 0.7976
Epoch 13/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5744 - acc: 0.8041 - val_loss: 0.6063 - val_acc: 0.7987
Epoch 14/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5462 - acc: 0.8137 - val_loss: 0.5972 - val_acc: 0.8016
Epoch 15/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5232 - acc: 0.8216 - val_loss: 0.6160 - val_acc: 0.7992
Epoch 16/50
390/390 [==============================] - 20s 52ms/step - loss: 0.5200 - acc: 0.8219 - val_loss: 0.6113 - val_acc: 0.8012
Epoch 17/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4960 - acc: 0.8295 - val_loss: 0.6038 - val_acc: 0.8067
Epoch 18/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4945 - acc: 0.8319 - val_loss: 0.6057 - val_acc: 0.8049
Epoch 19/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4738 - acc: 0.8377 - val_loss: 0.5847 - val_acc: 0.8098
Epoch 20/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4645 - acc: 0.8417 - val_loss: 0.5949 - val_acc: 0.8087
Epoch 21/50
390/390 [==============================] - 20s 51ms/step - loss: 0.4583 - acc: 0.8436 - val_loss: 0.5982 - val_acc: 0.8061
Epoch 22/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4498 - acc: 0.8464 - val_loss: 0.6152 - val_acc: 0.8091
Epoch 23/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4370 - acc: 0.8520 - val_loss: 0.5860 - val_acc: 0.8156
Epoch 24/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4305 - acc: 0.8533 - val_loss: 0.5713 - val_acc: 0.8161
Epoch 25/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4310 - acc: 0.8525 - val_loss: 0.6152 - val_acc: 0.8049
Epoch 26/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4141 - acc: 0.8595 - val_loss: 0.5789 - val_acc: 0.8178
Epoch 27/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4101 - acc: 0.8597 - val_loss: 0.6120 - val_acc: 0.8190
Epoch 28/50
390/390 [==============================] - 21s 53ms/step - loss: 0.4150 - acc: 0.8590 - val_loss: 0.6116 - val_acc: 0.8070
Epoch 29/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3986 - acc: 0.8662 - val_loss: 0.5939 - val_acc: 0.8134
Epoch 30/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3879 - acc: 0.8679 - val_loss: 0.5760 - val_acc: 0.8234
Epoch 31/50
390/390 [==============================] - 20s 52ms/step - loss: 0.4015 - acc: 0.8648 - val_loss: 0.5806 - val_acc: 0.8163
Epoch 32/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3828 - acc: 0.8696 - val_loss: 0.6067 - val_acc: 0.8192
Epoch 33/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3763 - acc: 0.8719 - val_loss: 0.6054 - val_acc: 0.8135
Epoch 34/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3763 - acc: 0.8725 - val_loss: 0.5725 - val_acc: 0.8231
Epoch 35/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3740 - acc: 0.8725 - val_loss: 0.6352 - val_acc: 0.8083
Epoch 36/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3644 - acc: 0.8762 - val_loss: 0.6205 - val_acc: 0.8183
Epoch 37/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3659 - acc: 0.8760 - val_loss: 0.5732 - val_acc: 0.8233
Epoch 38/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3587 - acc: 0.8797 - val_loss: 0.5949 - val_acc: 0.8260
Epoch 39/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3461 - acc: 0.8818 - val_loss: 0.5878 - val_acc: 0.8258
Epoch 40/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3486 - acc: 0.8819 - val_loss: 0.5575 - val_acc: 0.8336
Epoch 41/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3471 - acc: 0.8827 - val_loss: 0.6065 - val_acc: 0.8207
Epoch 42/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3416 - acc: 0.8861 - val_loss: 0.5915 - val_acc: 0.8206
Epoch 43/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3348 - acc: 0.8880 - val_loss: 0.5904 - val_acc: 0.8301
Epoch 44/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3410 - acc: 0.8870 - val_loss: 0.5996 - val_acc: 0.8234
Epoch 45/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3375 - acc: 0.8871 - val_loss: 0.5951 - val_acc: 0.8223
Epoch 46/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3323 - acc: 0.8890 - val_loss: 0.6227 - val_acc: 0.8239
Epoch 47/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3348 - acc: 0.8888 - val_loss: 0.6007 - val_acc: 0.8263
Epoch 48/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3260 - acc: 0.8913 - val_loss: 0.5829 - val_acc: 0.8317
Epoch 49/50
390/390 [==============================] - 20s 51ms/step - loss: 0.3193 - acc: 0.8934 - val_loss: 0.5959 - val_acc: 0.8304
Epoch 50/50
390/390 [==============================] - 20s 52ms/step - loss: 0.3231 - acc: 0.8924 - val_loss: 0.5765 - val_acc: 0.8287
Model took 1018.45 seconds to train

Accuracy on test data is: 82.87



### Logs for model using Separable Covolution Layer :

Epoch 1/49

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
  1/390 [..............................] - ETA: 56s - loss: 0.6633 - acc: 0.7422

/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:24: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:24: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., callbacks=[<keras.ca..., validation_data=(array([[[..., verbose=1, steps_per_epoch=390, epochs=49)`

390/390 [==============================] - 42s 107ms/step - loss: 0.9235 - acc: 0.6775 - val_loss: 0.8598 - val_acc: 0.7104
Epoch 2/49

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
390/390 [==============================] - 41s 105ms/step - loss: 0.8159 - acc: 0.7149 - val_loss: 0.8102 - val_acc: 0.7286
Epoch 3/49

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
390/390 [==============================] - 40s 104ms/step - loss: 0.7623 - acc: 0.7318 - val_loss: 0.8633 - val_acc: 0.7082
Epoch 4/49

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
390/390 [==============================] - 40s 104ms/step - loss: 0.7233 - acc: 0.7465 - val_loss: 0.6895 - val_acc: 0.7646
Epoch 5/49

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
390/390 [==============================] - 40s 103ms/step - loss: 0.6948 - acc: 0.7605 - val_loss: 0.7401 - val_acc: 0.7512
Epoch 6/49

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
390/390 [==============================] - 40s 103ms/step - loss: 0.6680 - acc: 0.7669 - val_loss: 0.6886 - val_acc: 0.7666
Epoch 7/49

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
390/390 [==============================] - 40s 102ms/step - loss: 0.6444 - acc: 0.7762 - val_loss: 0.6238 - val_acc: 0.7861
Epoch 8/49

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
390/390 [==============================] - 42s 107ms/step - loss: 0.6247 - acc: 0.7817 - val_loss: 0.6603 - val_acc: 0.7774
Epoch 9/49

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
390/390 [==============================] - 41s 105ms/step - loss: 0.6168 - acc: 0.7837 - val_loss: 0.6895 - val_acc: 0.7733
Epoch 10/49

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
390/390 [==============================] - 41s 104ms/step - loss: 0.5970 - acc: 0.7926 - val_loss: 0.6193 - val_acc: 0.7906
Epoch 11/49

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
390/390 [==============================] - 40s 104ms/step - loss: 0.5884 - acc: 0.7958 - val_loss: 0.6115 - val_acc: 0.7950
Epoch 12/49

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
390/390 [==============================] - 40s 103ms/step - loss: 0.5704 - acc: 0.8016 - val_loss: 0.6469 - val_acc: 0.7901
Epoch 13/49

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
390/390 [==============================] - 40s 102ms/step - loss: 0.5581 - acc: 0.8047 - val_loss: 0.5648 - val_acc: 0.8112
Epoch 14/49

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
390/390 [==============================] - 41s 105ms/step - loss: 0.5574 - acc: 0.8033 - val_loss: 0.5985 - val_acc: 0.8011
Epoch 15/49

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
390/390 [==============================] - 41s 105ms/step - loss: 0.5453 - acc: 0.8084 - val_loss: 0.5729 - val_acc: 0.8098
Epoch 16/49

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
390/390 [==============================] - 41s 104ms/step - loss: 0.5336 - acc: 0.8137 - val_loss: 0.5798 - val_acc: 0.8050
Epoch 17/49

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
390/390 [==============================] - 41s 104ms/step - loss: 0.5276 - acc: 0.8173 - val_loss: 0.5616 - val_acc: 0.8145
Epoch 18/49

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
390/390 [==============================] - 41s 104ms/step - loss: 0.5139 - acc: 0.8207 - val_loss: 0.5890 - val_acc: 0.8089
Epoch 19/49

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
390/390 [==============================] - 41s 104ms/step - loss: 0.5141 - acc: 0.8201 - val_loss: 0.5036 - val_acc: 0.8283
Epoch 20/49

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
390/390 [==============================] - 41s 105ms/step - loss: 0.5085 - acc: 0.8211 - val_loss: 0.5518 - val_acc: 0.8186
Epoch 21/49

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004065041.
390/390 [==============================] - 41s 105ms/step - loss: 0.4969 - acc: 0.8262 - val_loss: 0.5400 - val_acc: 0.8196
Epoch 22/49

Epoch 00022: LearningRateScheduler setting learning rate to 0.000389661.
390/390 [==============================] - 41s 105ms/step - loss: 0.4957 - acc: 0.8257 - val_loss: 0.5206 - val_acc: 0.8244
Epoch 23/49

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003741581.
390/390 [==============================] - 40s 104ms/step - loss: 0.4906 - acc: 0.8273 - val_loss: 0.5123 - val_acc: 0.8302
Epoch 24/49

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003598417.
390/390 [==============================] - 40s 103ms/step - loss: 0.4837 - acc: 0.8292 - val_loss: 0.4995 - val_acc: 0.8339
Epoch 25/49

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003465804.
390/390 [==============================] - 40s 103ms/step - loss: 0.4834 - acc: 0.8291 - val_loss: 0.5305 - val_acc: 0.8262
Epoch 26/49

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003342618.
390/390 [==============================] - 41s 105ms/step - loss: 0.4863 - acc: 0.8296 - val_loss: 0.5279 - val_acc: 0.8267
Epoch 27/49

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003227889.
390/390 [==============================] - 41s 104ms/step - loss: 0.4747 - acc: 0.8332 - val_loss: 0.5332 - val_acc: 0.8235
Epoch 28/49

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003120774.
390/390 [==============================] - 41s 104ms/step - loss: 0.4685 - acc: 0.8350 - val_loss: 0.5256 - val_acc: 0.8250
Epoch 29/49

Epoch 00029: LearningRateScheduler setting learning rate to 0.000302054.
390/390 [==============================] - 40s 104ms/step - loss: 0.4696 - acc: 0.8359 - val_loss: 0.5189 - val_acc: 0.8292
Epoch 30/49

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002926544.
390/390 [==============================] - 41s 106ms/step - loss: 0.4654 - acc: 0.8360 - val_loss: 0.5032 - val_acc: 0.8322
Epoch 31/49

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002838221.
390/390 [==============================] - 41s 105ms/step - loss: 0.4623 - acc: 0.8370 - val_loss: 0.5141 - val_acc: 0.8317
Epoch 32/49

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002755074.
390/390 [==============================] - 41s 104ms/step - loss: 0.4591 - acc: 0.8392 - val_loss: 0.4953 - val_acc: 0.8362
Epoch 33/49

Epoch 00033: LearningRateScheduler setting learning rate to 0.000267666.
390/390 [==============================] - 40s 103ms/step - loss: 0.4567 - acc: 0.8404 - val_loss: 0.4896 - val_acc: 0.8370
Epoch 34/49

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002602585.
390/390 [==============================] - 41s 104ms/step - loss: 0.4558 - acc: 0.8395 - val_loss: 0.4983 - val_acc: 0.8349
Epoch 35/49

Epoch 00035: LearningRateScheduler setting learning rate to 0.00025325.
390/390 [==============================] - 41s 104ms/step - loss: 0.4501 - acc: 0.8421 - val_loss: 0.4904 - val_acc: 0.8384
Epoch 36/49

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002466091.
390/390 [==============================] - 40s 104ms/step - loss: 0.4502 - acc: 0.8425 - val_loss: 0.5108 - val_acc: 0.8332
Epoch 37/49

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002403076.
390/390 [==============================] - 41s 104ms/step - loss: 0.4459 - acc: 0.8434 - val_loss: 0.5059 - val_acc: 0.8341
Epoch 38/49

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002343201.
390/390 [==============================] - 41s 104ms/step - loss: 0.4441 - acc: 0.8445 - val_loss: 0.4870 - val_acc: 0.8392
Epoch 39/49

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002286237.
390/390 [==============================] - 41s 104ms/step - loss: 0.4417 - acc: 0.8439 - val_loss: 0.5027 - val_acc: 0.8353
Epoch 40/49

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002231977.
390/390 [==============================] - 41s 106ms/step - loss: 0.4331 - acc: 0.8474 - val_loss: 0.4965 - val_acc: 0.8373
Epoch 41/49

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002180233.
390/390 [==============================] - 41s 106ms/step - loss: 0.4377 - acc: 0.8463 - val_loss: 0.4866 - val_acc: 0.8398
Epoch 42/49

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002130833.
390/390 [==============================] - 41s 105ms/step - loss: 0.4361 - acc: 0.8466 - val_loss: 0.5197 - val_acc: 0.8291
Epoch 43/49

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002083623.
390/390 [==============================] - 41s 104ms/step - loss: 0.4276 - acc: 0.8502 - val_loss: 0.5039 - val_acc: 0.8363
Epoch 44/49

Epoch 00044: LearningRateScheduler setting learning rate to 0.0002038459.
390/390 [==============================] - 40s 104ms/step - loss: 0.4300 - acc: 0.8477 - val_loss: 0.5070 - val_acc: 0.8371
Epoch 45/49

Epoch 00045: LearningRateScheduler setting learning rate to 0.0001995211.
390/390 [==============================] - 41s 105ms/step - loss: 0.4266 - acc: 0.8503 - val_loss: 0.4829 - val_acc: 0.8418
Epoch 46/49

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001953761.
390/390 [==============================] - 41s 104ms/step - loss: 0.4294 - acc: 0.8492 - val_loss: 0.4898 - val_acc: 0.8423
Epoch 47/49

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001913998.
390/390 [==============================] - 41s 105ms/step - loss: 0.4228 - acc: 0.8514 - val_loss: 0.4780 - val_acc: 0.8418
Epoch 48/49

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001875821.
390/390 [==============================] - 41s 104ms/step - loss: 0.4209 - acc: 0.8531 - val_loss: 0.4952 - val_acc: 0.8382
Epoch 49/49

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001839137.
390/390 [==============================] - 41s 104ms/step - loss: 0.4200 - acc: 0.8537 - val_loss: 0.5095 - val_acc: 0.8331
Model took 1993.15 seconds to train

Accuracy on test data is: 83.31


