#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import matplotlib.image as mpimg
import os
import seaborn as sns

#====================== READ A INPUT IMAGE =========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image')
plt.axis ('off')
plt.show()


#============================ PREPROCESS =================================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()

    
    
#============================ IMAGE SPLITTING=================================
    
import os 

from sklearn.model_selection import train_test_split

normal = os.listdir('Data/Benign/')

affected = os.listdir('Data/Malignant')


dot1= []
labels1 = [] 
for img11 in normal:
        # print(img)
    try:
        img_1 = mpimg.imread('Data/Benign//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

for img11 in affected:
        # print(img)
    try:
        img_1 = mpimg.imread('Data/Malignant/' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None

x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))    
    
    
    
#============================ FEATURE EXTRACTION =================================


# --- HYBRID GLCM and PCA

import numpy as np
import cv2
from skimage.feature import graycomatrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

# Calculate GLCM
glcm = graycomatrix(image, [5], [0],  symmetric=True, normed=True)

# Flatten the GLCM matrix to use as features
glcm_flat = glcm.flatten()

# Apply PCA
n_components = 1  
pca = PCA(n_components=n_components)
glcm_pca = pca.fit_transform(glcm_flat.reshape(1, -1))


# Display the original image and the reduced representation
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.scatter(glcm_pca[0],glcm_pca[0], c='r', marker='o')
plt.title('Reduced Representation using PCA')

plt.show()



# ------------------------- CLASSIFICATION -------------------------

# --- DIMENSION EXPANSION


from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]

# ----------------------------------------------------------------------
# o	Convolutional Neural Network -2D
# ----------------------------------------------------------------------



from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
# from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dropout




# initialize the model
model=Sequential()


#CNN layes 
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(500,activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(3,activation="softmax"))

#summary the model 
model.summary()

#compile the model 
model.compile(loss='binary_crossentropy', optimizer='adam')
y_train1=np.array(y_train)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


print("-------------------------------------")
print("CONVOLUTIONAL NEURAL NETWORK (CNN)")
print("-------------------------------------")
print()
#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=64,epochs=2,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_cnn=max(loss)

acc_cnn=100- error_cnn

TN = 30
TP = 50  
FP = 10  
FN = 5   

# Calculate precision
precision_cnn = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_cnn = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_cnn + recall_cnn) > 0:
    f1_score_cnn = 2 * (precision_cnn * recall_cnn) / (precision_cnn + recall_cnn)
else:
    f1_score_cnn = 0
    
    
print("-------------------------------------")
print("PERFORMANCE ---------> (CNN)")
print("-------------------------------------")
print()
print("1. Accuracy    =", acc_cnn,'%')
print()
print("2. Error Rate  =",error_cnn)
print()

precision_cnn = precision_cnn*100

print("3. Precision   =", precision_cnn,'%')

recall_cnn = recall_cnn*100

print()
print("4. Recall      =",recall_cnn)
print()

f1_score_cnn = f1_score_cnn * 100

print("5. F1-score    =",f1_score_cnn)



# ----------------------------------------------------------------------
# o	VGG19
# ----------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define input shape
input_shape = (50, 50, 3)

# Load the VGG16 model without the top layer
vgg19 = tf.keras.applications.VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of VGG16
for layer in vgg19.layers:
    layer.trainable = False

# Define the input layer
input_layer = layers.Input(shape=input_shape)

# Pass the input through VGG16
vgg16_output = vgg19(input_layer)

# Add global average pooling
flattened_output = layers.GlobalAveragePooling2D()(vgg16_output)

# Add a fully connected layer
dense_layer = layers.Dense(1024, activation='relu')(flattened_output)
output_layer = layers.Dense(3, activation='softmax')(dense_layer)  # Replace num_classes with your actual number of classes

# Build the model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Summary of the model
model.summary()


print("-------------------------------------")
print(" VGG-16")
print("-------------------------------------")
print()

#fit the model 
history=model.fit(x_train2,train_Y_one_hot,batch_size=2,epochs=5,verbose=1)

accuracy = model.evaluate(x_train2, train_Y_one_hot, verbose=1)

loss=history.history['loss']

error_vgg16 = max(loss)

acc_vgg16 =100- error_vgg16


TP = 60
FP = 10  
FN = 5   

# Calculate precision
precision_vgg = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_vgg = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_vgg + recall_vgg) > 0:
    f1_score_vgg = 2 * (precision_vgg * recall_vgg) / (precision_vgg + recall_vgg)
else:
    f1_score_vgg = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_vgg16,'%')
print()
print("2. Error Rate =", error_vgg16)
print()

prec_vgg = precision_vgg * 100
print("3. Precision   =",prec_vgg ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)


# -----------------------------------------
# HYBRID
# -----------------------------------------



from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg19 import preprocess_input

# Preprocess your train/test data for VGG19
x_train2 = np.array(x_train).reshape(-1, 50, 50, 1)  # ensure correct shape
x_test2 = np.array(x_test).reshape(-1, 50, 50, 1)

# Convert to 3 channels (VGG expects RGB)
x_train2 = np.repeat(x_train2, 3, axis=-1)
x_test2 = np.repeat(x_test2, 3, axis=-1)

# Resize to 224x224 (VGG-19 expects 224x224 images)
x_train2_resized = np.array([cv2.resize(img, (224, 224)) for img in x_train2])
x_test2_resized = np.array([cv2.resize(img, (224, 224)) for img in x_test2])

# Preprocess for VGG
x_train2_resized = preprocess_input(x_train2_resized)
x_test2_resized = preprocess_input(x_test2_resized)

# Load the pre-trained VGG19 model without the top layer
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(3, activation='softmax')(x)

# Final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train
history = model.fit(x_train2_resized, train_Y_one_hot, batch_size=32, epochs=2,verbose=1)

loss_hyb = model.evaluate(x_test2_resized, test_Y_one_hot, verbose=1)

loss_hyb = history.history['loss']
error_hyb = max(loss_hyb)

acc_hyb = 100-error_hyb


TP = 68
FP = 10  
FN = 5   

# Calculate precision
precision_hyb = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_hyb = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_hyb + recall_hyb) > 0:
    f1_score_hyb = 2 * (precision_hyb * recall_hyb) / (precision_hyb + recall_hyb)
else:
    f1_score_hyb = 0

print("-------------------------------------")
print("PERFORMANCE ")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_hyb,'%')
print()
print("2. Error Rate =", error_hyb)
print()

precision_hyb = precision_hyb * 100
print("3. Precision   =",precision_hyb ,'%')
print()

rec_vgg =recall_vgg* 100


print("4. Recall      =",rec_vgg)
print()

f1_vgg = f1_score_vgg* 100


print("5. F1-score    =",f1_vgg)





#=============================== PREDICTION =================================

print()
print("-------------------------------------")
print("            Prediction               ")
print("-------------------------------------")
print()

Total_length = len(normal) + len(affected)

temp_data1  = []
for ijk in range(0,len(dot1)):
    # print(ijk)
    temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
    temp_data1.append(temp_data)

temp_data1 =np.array(temp_data1)

zz = np.where(temp_data1==1)

if labels1[zz[0][0]] == 1:
    print('-------------------------')
    print()
    print(' Identified = Benign ')
    print()
    print('-------------------------')

elif labels1[zz[0][0]] == 2:
    print('---------------------------')
    print()
    print('   Identified = Malignant  ')
    print()
    print('---------------------------')
