from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50


def gesture_net(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """

    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    x = Conv2D(32, (5, 5), strides=[3,3], padding='valid')(img_input)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), strides=[2,2], padding='valid')(x)
    x = Activation('relu')(x) 
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), strides=[1,1], padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (5, 5), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, (3, 3), strides=[1,1], padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=[2,2])(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    
    x = Dense(64)(x)
    x = Dropout(0.5)(x)
    x = Activation('relu')(x)
    x = Dense(output_dim)(x)
    x = Activation('softmax')(x)

    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    print(model.summary())

    return model
    

    
def resnet50(img_width, img_height, img_channels, output_dim):
    """
    Define model architecture.

    # Arguments
       img_width: Target image widht.
       img_height: Target image height.
       img_channels: Target image channels.
       output_dim: Dimension of model output.

    # Returns
       model: A Model instance.
    """
    
    # Input
    img_input = Input(shape=(img_height, img_width, img_channels))
    
    # ResNet50
    model = ResNet50(include_top=False, weights='imagenet', input_tensor=img_input)
    x = model.output
    
    # FC layers
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(output_dim)(x)
    x = Activation('softmax')(x)
    
    # Define steering-collision model
    model = Model(inputs=[img_input], outputs=[x])
    print(model.summary())

    return model
    


