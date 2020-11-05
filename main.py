from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math

#HIGHER IS BETTER
n=400
nosofmatrices=4
# Opening the image
img1 = Image.open('chehra/sampleimg/img1.jpg')
img2 = Image.open('chehra/sampleimg/img2.jpg')
img3 = Image.open('chehra/sampleimg/img3.jpg')
img4 = Image.open('chehra/sampleimg/img4.jpg')

# Resizing
resized_img1=img1.resize((n,n))
resized_img2=img2.resize((n,n))
resized_img3=img3.resize((n,n))
resized_img4=img4.resize((n,n))

# Converting to Grayscale
img1_2 = ImageOps.grayscale(resized_img1)
img2_2 = ImageOps.grayscale(resized_img2)
img3_2 = ImageOps.grayscale(resized_img3)
img4_2 = ImageOps.grayscale(resized_img4)

#img2.show()

# Converting the image to NxN matrix
b1 = np.array(img1_2)
b2 = np.array(img2_2)
b3 = np.array(img3_2)
b4 = np.array(img4_2)

# Converting the NxN matrix to N^2x1 matrix
b1_new=b1.reshape((n*n,1))
b2_new=b2.reshape((n*n,1))
b3_new=b3.reshape((n*n,1))
b4_new=b4.reshape((n*n,1))

# Calculating the mean matrix
meanmatrix=(b1_new+b2_new+b3_new+b4_new)/nosofmatrices
meandisplay=(b1+b2+b3+b4)/nosofmatrices

#TODO:Use for loop instead of column_stack
b1_=b1_new-meanmatrix
b2_=b2_new-meanmatrix
b3_=b3_new-meanmatrix
b4_=b4_new-meanmatrix
aMatrix=np.column_stack((b1_,b2_,b3_,b4_))
#print(meanmatrix)

#print("A matrix: \n",aMatrix)


#Displays the Eigenfaced

# plt.imshow(b1new, interpolation='nearest')
# plt.show()
#img1.show()

plt.imshow(b1, cmap='gray',vmin=0, vmax=255)
#plt.show()
# plt.imshow(meanmatrix, cmap='gray',vmin=0, vmax=255)
# plt.show()
plt.imshow(meandisplay, cmap='gray',vmin=0, vmax=50)
#plt.show()

#Get transpose of aMatrix
p = nosofmatrices
q = n*n
aMatrix_T = np.zeros((p,q))
for i in range(0,p):
    for j in range(0,q):
        aMatrix_T[i][j] = aMatrix[j][i]
#print("TRANSPOSE: \n",aMatrix_T)

#A*A_t
p = nosofmatrices
q = n*n
multiply = np.zeros((nosofmatrices,nosofmatrices))
sum=0
for i in range(0,p):
           for j in range(0,p):
              for k in range (0,q):
                 sum = sum + aMatrix_T[i][k] * aMatrix[k][j]
              multiply[i][j] = sum
              sum=0
#print("A*A_T: \n",multiply)
#plt.imshow(multiply, interpolation='nearest')
#plt.show()

#SVD
AtA = multiply
n= nosofmatrices
d = np.zeros((nosofmatrices,nosofmatrices))
s = np.zeros((nosofmatrices,nosofmatrices))
s1 = np.zeros((nosofmatrices,nosofmatrices))
s1t = np.zeros((nosofmatrices,nosofmatrices))
temp = np.zeros((nosofmatrices,nosofmatrices))
zero= 1e-4
pi = 3.141592654
#zero=1e-4
for i in range (0,n):
 for j  in range(0,n):
   d[i][j]=AtA[i][j]
   s[i][j]=0

for i in range(0,n):
   s[i][i]=1
#do while loop in py
flag=0
i=0
j=1
max=math.fabs(d[0][1])
for p in range(0,n):
   for q in range(0,n):
      if(p!=q):
         if(max < math.fabs(d[p][q])):
            max = math.fabs(d[p][q])
            i=p
            j=q
if(d[i][i]==d[j][j]):
   if(d[i][j] > 0): 
      theta=pi/4 
   else: 
      theta=-pi/4
else:
   theta=0.5*math.atan(2*d[i][j]/(d[i][i]-d[j][j]))
for p in range(0,n):
   for q in range(0,n):
      s1[p][q]=0
      s1t[p][q]=0
for p in range(0,n):
   s1[p][p]=1
   s1t[p][p]=1
s1[i][i]=math.cos(theta)
s1[j][j]=s1[i][i]
s1[j][i]=math.sin(theta)
s1[i][j]=-s1[j][i]
s1t[i][i]=s1[i][i]
s1t[j][j]=s1[j][j]
s1t[i][j]=s1[j][i]
s1t[j][i]=s1[i][j]

for i in range(0,n):
   for j in range(0,n):
      temp[i][j]=0
      for p in range(0,n):
         temp[i][j]+=s1t[i][p]*d[p][j]

for i in range(0,n):
   for j in range(0,n):
      d[i][j]=0
      for p in range(0,n): 
         d[i][j]+=temp[i][p]*s1[p][j]
   
for i in range(0,n):
   for j in range(0,n):
      temp[i][j]=0
      for p in range(0,n):
         temp[i][j]+=s[i][p]*s1[p][j]

for i in range(0,n):
   for j in range(0,n):
      s[i][j]=temp[i][j]
for i in range(0,n):
   for j in range(0,n):
      if(i!=j):
         if(math.fabs(d[i][j] > zero)):
            flag=1
while(flag==1):
   flag=0
   i=0
   j=1
   max=math.fabs(d[0][1])
   for p in range(0,n):
      for q in range(0,n):
         if(p!=q):
            if(max < math.fabs(d[p][q])):
               max = math.fabs(d[p][q])
               i=p
               j=q
   if(d[i][i]==d[j][j]):
      if(d[i][j] > 0): 
         theta=pi/4 
      else: 
         theta=-pi/4
   else:
      theta=0.5*math.atan(2*d[i][j]/(d[i][i]-d[j][j]))
   for p in range(0,n):
      for q in range(0,n):
         s1[p][q]=0
         s1t[p][q]=0
   for p in range(0,n):
      s1[p][p]=1
      s1t[p][p]=1
   s1[i][i]=math.cos(theta)
   s1[j][j]=s1[i][i]
   s1[j][i]=math.sin(theta)
   s1[i][j]=-s1[j][i]
   s1t[i][i]=s1[i][i]
   s1t[j][j]=s1[j][j]
   s1t[i][j]=s1[j][i]
   s1t[j][i]=s1[i][j]

   for i in range(0,n):
      for j in range(0,n):
         temp[i][j]=0
         for p in range(0,n):
            temp[i][j]+=s1t[i][p]*d[p][j]

   for i in range(0,n):
      for j in range(0,n):
         d[i][j]=0
         for p in range(0,n): 
            d[i][j]+=temp[i][p]*s1[p][j]
      
   for i in range(0,n):
      for j in range(0,n):
         temp[i][j]=0
         for p in range(0,n):
            temp[i][j]+=s[i][p]*s1[p][j]

   for i in range(0,n):
      for j in range(0,n):
         s[i][j]=temp[i][j]
   for i in range(0,n):
      for j in range(0,n):
         if(i!=j):
            if(math.fabs(d[i][j] > zero)):
               flag=1


print("The eigenvalues are \n")
for i in range(0,n):
   print(d[i][i])
print("\nThe corresponding eigenvectors are \n")
for j in range(0,n):
   print("Eigen Vector->", j+1)
   for i in range(0,n):
      print(s[i][j])
   print("\n")
