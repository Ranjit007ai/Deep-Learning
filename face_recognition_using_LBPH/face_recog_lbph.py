# importing the nessesary libraries
import numpy as np 
import os
import cv2

# creating the list which will store the face_images and their correspounind labels
faces_list = []
face_label_list = []

# providing the path of the directory in which our dataset is present
#***********Use have to provide the path of your own place *****
path = r'C:\Users\Ranjit\Desktop\dl_program\face_recog_lbph\dataset'

for subfolder in os.listdir(path):
    current_path = os.path.join(path,subfolder)
    for current_img in os.listdir(current_path):
        # reading the image and storing the image matrix in the img variable
        image = cv2.imread(os.path.join(current_path,current_img))
        # resizing image to (200*200)
        image = cv2.resize(image,(200,200))

        # converting the BGR image to Gray Scale
        gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
        # appending the gray image to the face_list
        faces_list.append(gray_image)
        # appending the corresponding label in the face_label_list
        face_label_list.append(int(subfolder))

print(len(faces_list))
print(len(face_label_list))
# Converting the list into numpy Array ,since it will be helpful in performing the operations on it.
faces_list = np.array(faces_list)
face_label_list = np.array(face_label_list)
print(faces_list.shape)
print(face_label_list.shape)


# Now Training our dataset using LBPH 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces_list,face_label_list)
# In order to save the model after Training
face_recognizer.save('face_recog_trained_model.yml')

# Creating the dictionary for the name of the person
names = {0:'Ranjit',
         1:'Priyank'}
         


