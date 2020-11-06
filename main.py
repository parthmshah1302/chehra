from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import math
   
#Declarations
n=400
nosofmatrices=20
unique_people = 5
similar_faces = 4

img = [[0 for x in range(similar_faces)] for y in range(unique_people)]

# Opening the image
for i in range(0,5):
   for j in range(0,4):
      img_path = "sampleimg/img"+ str(i+1)+ "." + str(j+1) + ".jpg"
      img[i][j] = Image.open(img_path)
      img[i][j] = img[i][j].resize((n,n))
        
# Converting to Grayscale
img_grey = [[0 for x in range(similar_faces)] for y in range(unique_people)]
for i in range(0,5):
   for j in range(0,4):
      img_grey[i][j] = ImageOps.grayscale(img[i][j])
      
# Converting the image to NxN matrix
b_1 = [[0 for x in range(similar_faces)] for y in range(unique_people)]
b = [[0 for x in range(similar_faces)] for y in range(unique_people)]
for i in range(0,unique_people):
   for j in range(0,similar_faces):
      b_1[i][j] = np.array(img_grey[i][j])
      plt.imshow(b_1[i][j], interpolation='nearest')
      #plt.show()

# Converting the NxN matrix to (N^2)x1 matrix
for i in range(0,unique_people):
   for j in range(0,similar_faces):
      b[i][j]=b_1[i][j].reshape((n*n,1))

# The dataset is put into a single matrix A, which would be decomposed
#TODO Use a loop here instead of manual input
aMatrix=np.column_stack((b[0][0],b[0][1],b[0][2],b[0][3],b[1][0],b[1][1],b[1][2],b[1][3],b[2][0],b[2][1],b[2][2],b[2][3],b[3][0],b[3][1],b[3][2],b[3][3],b[4][0],b[4][1],b[4][2],b[4][3]))
# print(aMatrix)

# Calculating the Mean Matrix
bMean=[[0 for x in range(n*n)] for y in range(unique_people)]
for i in range(0,unique_people):
   bMean[i]=0
   for j in range(0,similar_faces):
      bMean[i]+=(b[i][j])/similar_faces
# print(bMean[0])

#Average faces of each individual 
# TODO for i in range (0,unique_people):
#    b1mean=bMean[i].reshape([n,n],order='F')
#    b1meanImg=Image.fromarray(b1mean)
#    b1meanImg.show()

#Calculating the transpose of aMatrix
p = nosofmatrices
q = n*n
aMatrix_T = np.zeros((p,q))
for i in range(0,p):
    for j in range(0,q):
        aMatrix_T[i][j] = aMatrix[j][i]
print("ORIGINAL MATRIX IS: \n",aMatrix)         
print("TRANSPOSE MATRIX IS: \n",aMatrix_T)         

#Covariance Matrix=A_t*A
p = nosofmatrices
q = n*n
covarianceMat = np.zeros((nosofmatrices,nosofmatrices))
sum=0
for i in range(0,p):
   for j in range(0,p):
      for k in range (0,q):
         sum = sum + aMatrix_T[i][k] * aMatrix[k][j]
      covarianceMat[i][j] = sum
      sum=0
#print("A_T*A: \n",covarianceMat)
#plt.imshow(covarianceMat, interpolation='nearest')
#plt.show()

#Decomposing the Covariance using SVD
AtA = covarianceMat
n= nosofmatrices
d = np.zeros((nosofmatrices,nosofmatrices))
s = np.zeros((nosofmatrices,nosofmatrices))
s1 = np.zeros((nosofmatrices,nosofmatrices))
s1t = np.zeros((nosofmatrices,nosofmatrices))
temp = np.zeros((nosofmatrices,nosofmatrices))
zero= 1e-4
pi = 3.141592654
for i in range (0,n):
 for j  in range(0,n):
   d[i][j]=AtA[i][j]
   s[i][j]=0
#Converting s to an identity matrix 
for i in range(0,n):
   s[i][i]=1
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

sigmaMatrix = np.zeros((nosofmatrices,nosofmatrices))
print("eigen values are:")
for i in range(0,nosofmatrices):
   print(d[i][i])
   if(d[i][i]<0):
      d[i][i] *= (-1)
   sigmaMatrix[i][j] = math.sqrt(d[i][i])
   print(sigmaMatrix[i][j])

#print("The Sigma Matrix is ",sigmaMatrix)
# print("\nThe corresponding eigenvectors are \n")
# for j in range(0,n):
#    print("Eigen Vector->", j+1)
#    for i in range(0,n):
#       print(s[i][j])
#    print("\n")
V = np.zeros((nosofmatrices,nosofmatrices))
for i in range(0,nosofmatrices):
   for j in range(0,nosofmatrices):
      V[i][j] = s[i][j]
# print("V=", V)

#Calculating the Projections of each Unique face
bMeanT = [[0 for x in range(unique_people)] for y in range(n*n)]
projectionMatrix=[[0 for x in range(unique_people*similar_faces)] for y in range(1)]

# for i in range(0,unique_people):
#    bMean[i] = np.array(bMean[i])
#    p = 1
#    q = unique_people*similar_faces
#    bMeanT = np.zeros((p,q))
# print(p, q, bMean[0])
for i in range(0, unique_people):
   bMean[i] = np.asarray(bMean[i])

print('sjdfn\n', type(bMean))

for i in range(0,p):
   for j in range(0,q):
      print(i," ",j)
      bMeanT[i][j] = bMean[j][i]
   print(bMean[i].shape)
   print(V.shape)
   projectionMatrix[i]=np.matmul(bMeanT,V)
print("The projections are: ",projectionMatrix)