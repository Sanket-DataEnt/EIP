# Logs of 20 epochs with their training and validation accuracy.



Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 20s 341us/step - loss: 0.5358 - acc: 0.8497 - val_loss: 0.1234 - val_acc: 0.9773
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 12s 201us/step - loss: 0.2551 - acc: 0.9235 - val_loss: 0.0707 - val_acc: 0.9866
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 12s 201us/step - loss: 0.2030 - acc: 0.9384 - val_loss: 0.0607 - val_acc: 0.9854
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 12s 203us/step - loss: 0.1731 - acc: 0.9450 - val_loss: 0.0365 - val_acc: 0.9913
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 13s 219us/step - loss: 0.1554 - acc: 0.9485 - val_loss: 0.0370 - val_acc: 0.9904
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 14s 227us/step - loss: 0.1440 - acc: 0.9495 - val_loss: 0.0323 - val_acc: 0.9919
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 14s 228us/step - loss: 0.1344 - acc: 0.9521 - val_loss: 0.0307 - val_acc: 0.9911
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 14s 239us/step - loss: 0.1257 - acc: 0.9536 - val_loss: 0.0256 - val_acc: 0.9929
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 14s 231us/step - loss: 0.1185 - acc: 0.9542 - val_loss: 0.0285 - val_acc: 0.9913
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 12s 198us/step - loss: 0.1191 - acc: 0.9540 - val_loss: 0.0242 - val_acc: 0.9934
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 12s 204us/step - loss: 0.1115 - acc: 0.9567 - val_loss: 0.0218 - val_acc: 0.9938
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 12s 195us/step - loss: 0.1084 - acc: 0.9555 - val_loss: 0.0239 - val_acc: 0.9930
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 12s 196us/step - loss: 0.1058 - acc: 0.9571 - val_loss: 0.0216 - val_acc: 0.9939
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 12s 194us/step - loss: 0.1045 - acc: 0.9572 - val_loss: 0.0212 - val_acc: 0.9940
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 12s 202us/step - loss: 0.1022 - acc: 0.9561 - val_loss: 0.0228 - val_acc: 0.9933
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 12s 194us/step - loss: 0.1009 - acc: 0.9567 - val_loss: 0.0202 - val_acc: 0.9938
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 12s 197us/step - loss: 0.0998 - acc: 0.9570 - val_loss: 0.0213 - val_acc: 0.9932
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 12s 192us/step - loss: 0.0982 - acc: 0.9577 - val_loss: 0.0217 - val_acc: 0.9929
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 12s 194us/step - loss: 0.0946 - acc: 0.9575 - val_loss: 0.0195 - val_acc: 0.9949
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 12s 192us/step - loss: 0.0956 - acc: 0.9582 - val_loss: 0.0192 - val_acc: 0.9942

<keras.callbacks.History at 0x7fc6756dd978>


## result of your model.evaluate (on test data) : 

[0.019174820344615727, 0.9942]


## Strategy taken:

Reduced the parameters by reducing kernels.