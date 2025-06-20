import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import math
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# Preprocess the input data
def preprocess_image(image_path, target_size):
    """
    Preprocess the input image to match the model's input format.
    Arguments:
    - image_path: Path to the image file.
    - target_size: The target size that the model expects (e.g., (300, 300) or (224, 224)).
    
    Returns:
    - preprocessed image array ready for prediction.
    """
    image = Image.open(image_path).convert("RGB")
    image = T.Resize(target_size)(image)
    image = T.ToTensor()(image)
    return image
    
def load_challenge_data(filename, open_image=True):
    if open_image:
        data = preprocess_image(filename, target_size=(300, 300))
    else:
        data = None
    new_file = filename.replace('.png','.hea')
    input_header_file = os.path.join(new_file)
    with open(input_header_file,'r') as f:
        header_data=f.readlines()
    return data, header_data

def import_key_data(path):
    gender=[]
    age=[]
    labels=[]
    ecg_filenames=[]
    for subdir, dirs, files in sorted(os.walk(path)):
        for filename in files:
            filepath = subdir + os.sep + filename
            if filepath.endswith(".png"):
                # print(filepath)
                _, header_data = load_challenge_data(filepath, open_image=False)
                #get the line like '#Dx: 251146004,426783006' and extracts the ids substring only
                labels.append(header_data[15][5:-1])
                ecg_filenames.append(filepath)
                gender.append(header_data[14][6:-1])
                age.append(header_data[13][6:-1])
    return gender, age, labels, ecg_filenames
        
def make_undefined_class(labels, df_unscored):
    df_labels = pd.DataFrame(labels)
    for i in range(len(df_unscored.iloc[0:,1])):
        df_labels.replace(to_replace=str(df_unscored.iloc[i,1]), inplace=True ,value="undefined class", regex=True)

    #equivalent classes
    codes_to_replace=['713427006', '427172004']
    replace_with = ['59118001', '17338001']

    for i in range(len(codes_to_replace)):
        df_labels.replace(to_replace=codes_to_replace[i], inplace=True ,value=replace_with[i], regex=True)
    
    return df_labels

def onehot_encode(df_labels):
    one_hot = MultiLabelBinarizer()
    # print(df_labels[0].str.split(pat=','))
    y=one_hot.fit_transform(df_labels[0].str.split(pat=','))
    print("The classes we will look at are encoded as SNOMED CT codes:")
    print(one_hot.classes_)
    y = np.delete(y, -1, axis=1)
    print("classes: {}".format(y.shape[1]))
    return y, one_hot.classes_[0:-1]

def plot_classes(classes, scored_classes,y):
    for j in range(len(classes)):
        for i in range(len(scored_classes.iloc[:,1])):
            if (str(scored_classes.iloc[:,1][i]) == classes[j]):
                classes[j] = scored_classes.iloc[:,0][i]
    plt.figure(figsize=(30,20))
    plt.bar(x=classes,height=y.sum(axis=0))
    plt.title("Distribution of Diagnosis", color = "black", fontsize = 30)
    plt.tick_params(axis="both", colors = "black")
    plt.xlabel("Diagnosis", color = "black")
    plt.ylabel("Count", color = "black")
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize = 20)
    plt.savefig("fordeling.png")
    plt.show()


