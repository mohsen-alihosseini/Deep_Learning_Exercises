import pattern
import generator
import matplotlib.pyplot as plt

#_________Checkboard
Checkerboard=pattern.Checker(200,10)
Checkerboard.draw()
Checkerboard.show()


#_________Cirlce
circle = pattern.Circle(500,100,(250,250))
circle.draw()
circle.show()

#_________Spectrum
spect = pattern.Spectrum(100)
spect.draw()
spect.show()

#_________Generator
g = generator.ImageGenerator('./exercise_data/', './Labels.json', 12, [128, 128, 3], rotation=True, mirroring=True, shuffle=False)

g.next()
g.show()
# print(g.batch_number)

# g.next()
# g.show()
# print(g.batch_number)


# images, labels, batch_info, batch_num, epoch = g.next()
# for image_name, label in batch_info:

#     print(
#         f"Image: {image_name} | "
#         f"Class: {g.class_name(label)} | "
#         f"Batch: {batch_num} | "
#         f"Epoch: {epoch}"
#     )

# g.next()