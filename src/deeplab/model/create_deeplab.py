#-----------------------------------------------------
#Import external libraries
import numpy
import tensorflow
import horovod.tensorflow.keras as hvd
import keras
from keras.applications import ResNet50
from keras import layers
from keras import ops
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils import set_random_seed
set_random_seed(1234567) #to ensure reproducibility

#References:
#This code is adopted from S. Rakshit (2024): https://keras.io/examples/vision/deeplabv3_plus
#-----------------------------------------------------


#Function to define convolutional block
def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False):
    
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)

    return ops.nn.relu(x)


#Function to define pyramid pooling
def DilatedSpatialPyramidPooling(dspp_input):

    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)

    return output


#Function to define DeepLab, ideally for 512 x 512
def DeeplabV3Plus(image_size, num_classes):

    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.resnet50.preprocess_input(model_input)
    resnet50 = ResNet50(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)

    return keras.Model(inputs=model_input, outputs=model_output)


#Function to create unet --changes
def create_deeplabv3plus(training_images, training_labels, 
                         num_classes=4,
                         learning_rate=1E-3,
                         distributed=False,
                         info=False):
    
    #Set horovod
    if(distributed==True):
        hvd.init()

        gpus = tensorflow.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tensorflow.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    #Compile model
    model = DeeplabV3Plus(image_size=training_images.shape[1], num_classes=num_classes)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    if(distributed==True):
        optt  = Adam(learning_rate=learning_rate * hvd.size())
        optt  = hvd.DistributedOptimizer(optt)
    else:
        optt  = Adam(learning_rate=learning_rate)  

    model.compile(loss=loss, optimizer=optt)

    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())
        
    #Return model
    return model


#Function to load existing unet 
def load_existing_deeplabv3plus(kerasname=None, info=False):

    #Load the whole model in .keras file format
    model = load_model(kerasname, compile=False)

    #Display
    if(info==True):
        print("This is your model's architecture:")
        print(model.summary())
    
    #Return model
    return model