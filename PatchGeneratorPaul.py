import numpy as np
import cv2, os, random

class PatchGenerator(object):

    def __init__(self, input_dir, class_labels, batch_size, batches, augmentation_fn=None):

        # params
        self.input_dir       = input_dir       # path to patches in glob format
        self.class_labels    = class_labels    # list containing the classes
        self.batch_size      = batch_size      # number of patches per batch
        self.batches         = batches
                    
        #get directories to label and color
        self.batch_colors = [ sorted(os.listdir(os.path.join(self.input_dir, "train_color"+str(batch)))) for batch in self.batches ]
        self.batch_labels = [ sorted(os.listdir(os.path.join(self.input_dir, "train_label"+str(batch)))) for batch in self.batches ]
        
        lengths = [ len(batch) for batch in self.batch_colors ]
        self.totalLength = sum(lengths)
        self.percentages = [ batch/self.totalLength for batch in lengths ]
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        
        #loop until a batch is found
        batchnotfound = True
        while batchnotfound:  
            #pick random batch from the set of possible batches
            batch_color = np.random.choice(self.batch_colors, p = self.percentages)
            index = self.batch_colors.index(batch_color)
            batch_label = self.batch_labels[index]
            randomBatch = self.batches[index]
            
            #if there are not enough imges in the batch remove it from the list of possible batches
            if  len(batch_color) < self.batch_size:
                self.batch_colors.remove(batch_color)
                self.batch_labels.remove(batch_label)
            else:
                #take paired random samples form the available imges in the random batch foulder
                batch_x_names, batch_y_names = zip(*random.sample(list(zip(batch_color, batch_label)), self.batch_size))
                batchnotfound = False
                                     
        #ini the batches    
        batch_x = []
        batch_y = []
        #loop through all direcotries to the wanted images
        for i in range(len(batch_x_names)):
            # load the image file
            img = np.array(cv2.imread(os.path.join(self.input_dir, 'train_color'+str(randomBatch), batch_x_names[i]), -1))/255
            
            # load the label file
            labels_np = np.array(cv2.imread(os.path.join(self.input_dir, 'train_label'+str(randomBatch), batch_y_names[i]), -1))//1000
            
            labels_np[np.logical_not(np.isin(labels_np, self.class_labels))] = 0
            
            # encode in one_hot
            labels = (self.class_labels == labels_np[...,None]).astype(int)
            
            # store the image and the labels
            batch_x.append(img)
            batch_y.append(labels)
        
        #return the batches as np array
        return np.array(batch_x), np.array(batch_y)