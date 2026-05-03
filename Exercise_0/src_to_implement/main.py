import pattern
import generator
import matplotlib.pyplot as plt

### Checker debugging
a = pattern.Checker(80,10)
cc = a.draw()
a.show()
print(cc.shape)
c = pattern.Checker(500, 1)
res = c.draw()
res[:] = 0
print(res)
print(c.output)


### Cirlce debugging
a = pattern.Circle(500,100,(250,250))
a.draw()
a.show()


### Spectrum debugging
s = pattern.Spectrum(100)
s.draw()
s.show()


### Generator debugging
g = generator.ImageGenerator('./exercise_data/', './Labels.json', 3, [64, 64, 3], rotation=False, mirroring=False, shuffle=False)

g.next()


g.show()


# images, labels = g.next()
# for i,j in enumerate(images):
#     plt.imshow(j)
#     print(labels[i])
#     plt.show()