from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
from Model import build_cnn1


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
def train(data_set, batch_size =2000, epochs = 5):
    x_train,Y = data_set
    model_cnn.fit(x_train,Y, batch_size,epochs)



def test():
    global model_cnn
    digits_typies = 10
    input_shape = (28,28)
    model_cnn = build_cnn1((input_shape[0],input_shape[1],1),digits_typies)
    #loading data
    training_data, test_data = get_datas(input_shape,digits_typies)
    #train model
    train(training_data)
    #testing accuracy
    x_test, y_test = test_data
    predictions = model_cnn.predict(x_test)
    currect = 0
    for i in range(x_test.shape[0]):
         label = y_test[i]
         #pick up most likely number
         prediction = np.argmax(predictions[i])
         if prediction == int(label):
            currect += 1
    print("accuracy of my cnn in mnist test sets is %f"%(currect/x_test.shape[0]))


def main():
    test()

main()