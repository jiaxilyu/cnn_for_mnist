from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
from Model import build_cnn
import matplotlib.pyplot as plt
from tensorflow.keras import initializers


model_cnn = None 
#return the restructed training data set and test data set
def get_datas(pic_size, digits_typies):
    #reshape input data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape([-1,pic_size[0],pic_size[1],1])
    x_test = x_test.reshape([-1,pic_size[0],pic_size[1],1])
    #construct target/label vector
    Y = np.zeros((digits_typies, x_train.shape[0]),dtype=np.float64)
    for i in range(x_train.shape[0]):
        Y[y_train[i],i] = 1
    Y = Y.T
    return (x_train, Y),(x_test, y_test)

#train model
def train(model,data_set, batch_size =2000, epochs = 5):
    x_train,Y = data_set
    history = model.fit(x_train,Y, batch_size,epochs)
    return history.history['accuracy']

def get_accuracy(model, test_data):
    x_test, y_test = test_data
    predictions = model.predict(x_test)
    currect = 0
    for i in range(x_test.shape[0]):
         label = y_test[i]
         #pick up most likely number
         prediction = np.argmax(predictions[i])
         if prediction == int(label):
            currect += 1
    #print("accuracy of my cnn in mnist test sets is %f"%(currect/x_test.shape[0]))
    return currect/x_test.shape[0]

def plot_graph(accuracy_list, models_name):
    for accuracy in accuracy_list:
        plt.plot(accuracy)
    plt.legend(models_name, loc='upper left')
    plt.show()

def test():
    models = []
    models_name = []
    #diff model settings
    kernel_initializer1 = "he_uniform"
    kernel_initializer2 = "glorot_uniform"
    kernel_initializer3 = "random_normal"
    dense_units1 = 512
    dense_units2 = 1024
    kernels_list = [kernel_initializer1,kernel_initializer2,kernel_initializer3]
    dense_units_list = [dense_units1, dense_units2]
    digits_typies = 10
    input_shape = (28,28)
    #build diff models
    for kernel_initializer in kernels_list:
        for dense_units in dense_units_list:
            model_name = kernel_initializer + " + " + str(dense_units) + 'units'
            models_name.append(model_name)
            models.append(build_cnn(input_shape=(input_shape[0],input_shape[1],1),kernel_initializer =kernel_initializer,denes_layer_units=dense_units,output_size=digits_typies))
    #loading data
    training_data, test_data = get_datas(input_shape,digits_typies)
    #----------------------training and testing-----------------
    accuracy_list = []
    results = []
    #train model
    for model in models:
       accuracy_list.append(train(model,training_data))
       #testing accuracy
       results.append(get_accuracy(model, test_data))
    
    for result, model_name in zip(results,models_name):
        print("accuracy of %s in mnist test sets is %f"%(model_name,result))
    #plot graph
    plot_graph(accuracy_list, models_name)


def main():
    test()

main()