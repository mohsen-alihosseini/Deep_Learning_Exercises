import numpy as np
import matplotlib.pyplot as plt

class Checker():
    def __init__(self, resolution, tile_size):

        self.resolution = resolution
        self.tile_size = tile_size

        

    def draw(self):
        if self.resolution%(2*self.tile_size)==0 :
            # print("drawing")
            a_ = np.concatenate((np.zeros((self.tile_size,self.tile_size)), np.ones((self.tile_size,self.tile_size))), axis=1)
            b_ = 1-a_ #reverse the tiles
            tmp = np.concatenate((a_,b_),axis=0)
            reps = int(self.resolution / (2 * self.tile_size))
            self.output = np.tile(tmp,(reps,reps))
            # print(self.output)
        else:
            print("Givenvalue for Resulution is not dividable by 2")
        return np.array(self.output,copy=True)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle():
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        

    def draw(self):
        x= np.arange(self.resolution)
        y= np.arange(self.resolution)
        x,y=np.meshgrid(x,y)
        # print(self.position)
        c_x,c_y = self.position #circle center
        # print(x,y)
        self.output = np.zeros((self.resolution,self.resolution))   ##all zeroo
        self.output[(x-c_x)**2+(y-c_y)**2<self.radius**2]=1         ## all zero except where condition true
        # print(self.output)
        return np.array(self.output, copy=True)

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution,3))


    # Top-Left (Blue): Red=0, Green=0, Blue=1
    # Top-Right (Red): Red=1, Green=0, Blue=0
    # Bottom-Left (nili): Red=0, Green=1, Blue=1
    # Bottom-Right (Yellow): Red=1, Green=1, Blue=0

    # Red increases from left to right (x-axis)
    # Green increases from top to bottom (y-axis)
    # Blue decreases from left to right (x-axis).
    
    def draw(self):
        r_line= np.tile(np.linspace(0,1,self.resolution),self.resolution)
        self.output[:,:,0]=r_line.reshape((self.resolution,self.resolution))

        b_line= np.tile(np.linspace(1,0,self.resolution),self.resolution)
        self.output[:,:,2] =b_line.reshape((self.resolution,self.resolution))

        g_line= np.linspace(0,1,self.resolution).reshape(-1,1)
        g_line= np.tile(g_line,self.resolution)        
        self.output[:,:,1] = g_line

        return np.array(self.output, copy=True)


    def show(self):
        plt.imshow(self.output)
        plt.show()


