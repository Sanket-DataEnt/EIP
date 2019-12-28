# Result of Accuracies:

'''python
{'age_output_acc': 0.7764113122417081,
 'age_output_loss': 0.49176413974454325,
 'bag_output_acc': 0.7745295916834185,
 'bag_output_loss': 0.47757232958270657,
 'emotion_output_acc': 0.8577368951612904,
 'emotion_output_loss': 0.369359040452588,
 'footwear_output_acc': 0.976982545468115,
 'footwear_output_loss': 0.07685946387749526,
 'gender_output_acc': 0.9873991935483871,
 'gender_output_loss': 0.03953302520207099,
 'image_quality_output_acc': 0.9695900717089253,
 'image_quality_output_loss': 0.08420531833243947,
 'loss': 0.35118254442368785,
 'pose_output_acc': 0.8282930216481609,
 'pose_output_loss': 0.39288611662003303,
 'weight_output_acc': 0.8245967741935484,
 'weight_output_loss': 0.41082503141895416}

'''

# Architecture:

'''python
visible = Input(shape=(224,224,3))

#Block 1
conv1 = Conv2D(32, kernel_size=3, activation='relu',use_bias=False)(visible) #222
conv1_bn = BatchNormalization()(conv1)
conv1_d = Dropout(0.1)(conv1_bn)

#Block 2
conv2 = Conv2D(32, kernel_size=3, activation='relu',use_bias=False)(conv1_d) #220
conv2_bn = BatchNormalization()(conv2)
conv2_d = Dropout(0.1)(conv2_bn)


pool1 = MaxPooling2D(pool_size=(2, 2))(conv2_d) #110

#Block 3
conv3 = Conv2D(64, kernel_size=3, activation='relu',use_bias=False)(pool1) #108
conv3_bn = BatchNormalization()(conv3)
conv3_d = Dropout(0.1)(conv3_bn)

pool2 = MaxPooling2D(pool_size=(2, 2))(conv3_d) #54

#Block 4
conv4 = Conv2D(64, kernel_size=3, activation='relu',use_bias=False)(pool2) #52
conv4_bn = BatchNormalization()(conv4)
conv4_d = Dropout(0.1)(conv4_bn)

pool3= MaxPooling2D(pool_size=(2, 2))(conv4_d) #26

#Block 5
conv5 = Conv2D(128, kernel_size=3, activation='relu',use_bias=False)(pool3) #24
conv5_bn = BatchNormalization()(conv5)
conv5_d = Dropout(0.1)(conv5_bn)

pool4= MaxPooling2D(pool_size=(2, 2))(conv5_d) #12


neck = Conv2D(128, kernel_size=3, activation='relu')(pool4)
# neck = MaxPooling2D(pool_size=(2, 2))(conv2)


# neck = backbone.output
neck = Flatten(name="flatten")(neck)
neck = Dense(512, activation="relu")(neck)

# Define the model


def build_tower(in_layer):
    neck = Dropout(0.1)(in_layer) #0.5 to 0.1
    neck = Dense(128, activation="relu")(neck)
    #neck = Dropout(0.1)(in_layer) #0.5 to 0.1
    neck = Dense(128, activation="relu")(neck)
    return neck


def build_head(name, in_layer):
    return Dense(
        num_units[name], activation="softmax", name=f"{name}_output"
    )(in_layer)

# heads
gender = build_head("gender", build_tower(neck))
image_quality = build_head("image_quality", build_tower(neck))
age = build_head("age", build_tower(neck))
weight = build_head("weight", build_tower(neck))
bag = build_head("bag", build_tower(neck))
footwear = build_head("footwear", build_tower(neck))
emotion = build_head("emotion", build_tower(neck))
pose = build_head("pose", build_tower(neck))


model = Model(
    inputs=visible, 
    outputs=[gender, image_quality, age, weight, bag, footwear, pose, emotion]
)

'''

# Epochs :
I have saved 15 models and used approximately 300 epochs 