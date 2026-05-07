import pattern
import generator
import matplotlib.pyplot as plt

#_________Checkboard
# Checkerboard=pattern.Checker(200,10)
# Checkerboard.draw()
# Checkerboard.show()


#_________Cirlce
# circle = pattern.Circle(500,100,(250,250))
# circle.draw()
# circle.show()

#_________Spectrum
# spect = pattern.Spectrum(100)
# spect.draw()
# spect.show()

#_________Generator
g = generator.ImageGenerator('./exercise_data/', './Labels.json', 10, [128, 128, 3], rotation=True, mirroring=False, shuffle=False)

g.next()
g.show()
print(g.batch_number)
# g.next()
# g.show()
# print(g.batch_number)
