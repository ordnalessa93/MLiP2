import keras
import matplotlib.pyplot as plt
import numpy as np

class PrintData(keras.callbacks.Callback):
    
    def __init__(self):
        self.loss = []
        self.acc = []

    def on_epoch_end(self, epoch, logs={}):
        with open("current_status.txt", "a") as myfile:
            myfile.write(str(epoch) + ' ' + str(logs) + '\n')
        
        self.loss.append(logs['loss'])
        self.acc.append(logs['acc'])
        
        plt.plot(np.arange(epoch + 1), self.loss ,np.arange(epoch + 1), self.acc)
        plt.ylabel('epoch ' + str(epoch))
        plt.show()