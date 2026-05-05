# import os.path
# import json
# import scipy.misc
# import numpy as np
# import matplotlib.pyplot as plt
# import skimage

# # In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# # This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# # This input consists of a batch of images and its corresponding labels.
# class ImageGenerator:
#     def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
#         # Define all members of your generator class object as global members here.
#         # These need to include:
#         # the batch size
#         # the image size
#         # flags for different augmentations and whether the data should be shuffled for each epoch
#         # Also depending on the size of your data-set you can consider loading all images into memory here already.
#         # The labels are stored in json format and can be directly loaded as dictionary.
#         # Note that the file names correspond to the dicts of the label dictionary.

#         self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
#                            7: 'horse', 8: 'ship', 9: 'truck'}
#         #TODO: implement constructor
#         self.file_path = file_path
#         self.label_path = label_path
#         self.batch_size = batch_size
#         self.image_size = image_size
#         self.rotation = rotation
#         self.mirroring = mirroring
#         self.shuffle = shuffle

#         with open(self.label_path,'r') as f:
#             self.label_dict = json.load(f)
#         self.label_list = [(str(i),self.label_dict.get(str(i))) for i in sorted(list(map(int,self.label_dict.keys())))]
#         if shuffle:
#             np.random.shuffle(self.label_list)
#         self.batch_number = 0
#         self.epoch_number = 0
#         self.next_epoch = False       

#     def next(self):
#         # This function creates a batch of images and corresponding labels and returns them.
#         # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
#         # Note that your amount of total data might not be divisible without remainder with the batch_size.
#         # Think about how to handle such cases
#         #TODO: implement next method
        
#         #return images, labels
#         if self.next_epoch:
#                 self.epoch_number += 1
#                 self.next_epoch = False
#         output = self.label_list[(self.batch_size*self.batch_number): (self.batch_size*(self.batch_number+1))]
#         self.batch_number += 1
#         if self.batch_number == np.ceil((len(self.label_list)/self.batch_size)-0.001):
#                 output += self.label_list[:self.batch_size-len(output)]
#                 self.batch_number = 0
#                 self.next_epoch = True
#                 if self.shuffle:
#                     np.random.shuffle(self.label_list)
#                     # print(self.label_list)
#             # print('*'*2,output)
#             # if self.batch_number == 0: print('epoch ended')
#         images = np.zeros(tuple([self.batch_size]+self.image_size))
#         labels = []

#         for i,(im,l) in enumerate(output):
#             img = skimage.transform.resize(np.load(self.file_path+im+'.npy'), tuple(self.image_size))
#             if self.rotation or self.mirroring:
#                 img = self.augment(img)
#             images[i] = img
#             labels.append(l)
#         return images, labels

#     def augment(self,img):
#         # this function takes a single image as an input and performs a random transformation
#         # (mirroring and/or rotation) on it and outputs the transformed image
#         #TODO: implement augmentation function
#         if self.mirroring:
#             r = np.random.randint(0,3)
#             # print(r)
#             # plt.imshow(img)
#             # plt.show()
#             if r==0:
#                 img = img[::-1, :, :] # first dimension is flipped hence vertical mirroring
#             elif r==1:
#                 img = img[:, ::-1, :] # second dimension is flipped hence horizontal mirroring
#             # plt.imshow(img)
#             # plt.show()
#         if self.rotation:
#             r = np.random.randint(0,4)
#             # print(r)
#             # plt.imshow(img)
#             # plt.show()
#             for i in range(r):
#                 img = np.rot90(img)
#             # plt.imshow(img)
#             # plt.show()

#         return img

#     def current_epoch(self):
#         # return the current epoch number
#         return self.epoch_number

#     def class_name(self, x):
#         # This function returns the class name for a specific input
#         #TODO: implement class name function
#         return self.class_dict.get(x)
#     def show(self):
#         # In order to verify that the generator creates batches as required, this functions calls next to get a
#         # batch of images and labels and visualizes it.
#         #TODO: implement show method
#         images, labels = self.next()
#         fig = plt.figure()
#         for i, img in enumerate(images,1):
#             fig.add_subplot(int(np.ceil(self.batch_size/3)), 3, i, xticks=[], yticks=[], title=self.class_name(labels[i-1])).imshow(img)
#             # axss.set_title(self.class_name(labels[i]))
#         plt.show()


