from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.applications.resnet50 import ResNet50

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
    


