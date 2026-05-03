import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((resolution,resolution))
        # print(self.output.shape)

    def draw(self):
        if self.resolution%(2*self.tile_size)==0 :
            # print("drawing")
            temp = np.concatenate((np.zeros((self.tile_size,self.tile_size)), np.ones((self.tile_size,self.tile_size))), axis=1)
            temp = np.vstack((temp,1-temp))
            self.output = np.tile(temp,(int(((self.resolution/self.tile_size))/2), int(((self.resolution/self.tile_size))/2)))
            # self.output = temp.reshape((self.resolution,self.resolution))
            # print(self.output)
        return np.array(self.output,copy=True)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle():
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        # self.output = np.ones((self.resolution, self.resolution))

    def draw(self):
        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        x0,y0 = self.position
        # print(x,y)
        self.output = np.where((x-x0)**2+(y-y0)**2<self.radius**2,True,False)
        return np.array(self.output, copy=True)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution,3))

    def draw(self):
        self.output[:,:,0] = np.tile(np.linspace(0,1,self.resolution),self.resolution).reshape((self.resolution,self.resolution))
        self.output[:,:,2] = np.tile(np.linspace(1,0,self.resolution),self.resolution).reshape((self.resolution,self.resolution))
        self.output[:,:,1] = np.tile(np.linspace(0,1,self.resolution).reshape(-1,1),self.resolution)
        return np.array(self.output, copy=True)
        # x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        # self.output[:,:,2] = 1 - (x**2 + y**2)/(2*(self.resolution-1)**2)
        # self.output[:,:,1] = 1 - ((x-(self.resolution-1)/2)**2 + (y-(self.resolution-1))**2)/(2*(self.resolution-1)**2)
        # self.output[:,:,0] = 1 - ((x-(self.resolution-1))**2 + y**2)/(2*(self.resolution-1)**2)
        
        # self.output[:,:,0] = 1 - np.sqrt((x**2 + y**2)/(2*(self.resolution-1)**2))
        # self.output[:,:,1] = 1 - np.sqrt(((x-(self.resolution-1)/2)**2 + (y-(self.resolution-1))**2)/(2*(self.resolution-1)**2))
        # self.output[:,:,2] = 1 - np.sqrt(((x-(self.resolution-1))**2 + y**2)/(2*(self.resolution-1)**2))

        # self.output[:,:,2] = 1 - (x + y)/(self.resolution-1)
        # self.output[:,:,1] = 1 - (abs(x-(self.resolution-1)/2) + abs(y-(self.resolution-1)))/(self.resolution-1)
        # self.output[:,:,0] = 1 - (abs(x-(self.resolution-1)) + y)/(self.resolution-1)
        # self.output = np.where(self.output<0,0,self.output)
        # print(self.output[:,:,0])

    def show(self):
        # plt.imshow(self.output[:,:,1], cmap='gray')
        plt.imshow(self.output)
        plt.show()


# s = Spectrum(100)
# s.draw()
# s.show()

# a = Circle(100,25,(40,60))
# a.draw()
# a.show()

# a = Checker(80,10)
# cc = a.draw()
# a.show()
# print(cc.shape)
# c = Checker(100, 1)
# res = c.draw()
# res[:] = 0
# print(res)
# print(c.output)