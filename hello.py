from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

n=3

img1 = Image.open('sampleimg/img1.jpg')
img2 = Image.open('sampleimg/img2.jpg')
img3 = Image.open('sampleimg/img3.jpg')
img4 = Image.open('sampleimg/img4.jpg')
resized_img1=img1.resize((n,n))
resized_img2=img2.resize((n,n))
resized_img3=img3.resize((n,n))
resized_img4=img4.resize((n,n))

img1_2 = ImageOps.grayscale(resized_img1)
img2_2 = ImageOps.grayscale(resized_img2)
img3_2 = ImageOps.grayscale(resized_img3)
img4_2 = ImageOps.grayscale(resized_img4)

#img2.show()

b1 = np.array(img1_2)
b2 = np.array(img2_2)
b3 = np.array(img3_2)
b4 = np.array(img4_2)
b1_new=b1.reshape((n*n,1))
b2_new=b2.reshape((n*n,1))
b3_new=b3.reshape((n*n,1))
b4_new=b4.reshape((n*n,1))
meanmatrix=(b1+b2+b3+b4)/4
combined=np.concatenate([b1,b2,b3,b4],axis=1)
# print(meanmatrix)
print(combined)
b1new=b1-meanmatrix
b2new=b2-meanmatrix
b3new=b3-meanmatrix
b4new=b4-meanmatrix

# plt.imshow(b1new, interpolation='nearest')
# plt.show()
plt.imshow(combined, interpolation='nearest')
plt.show()
# plt.imshow(b1, interpolation='nearest')
# plt.show()

