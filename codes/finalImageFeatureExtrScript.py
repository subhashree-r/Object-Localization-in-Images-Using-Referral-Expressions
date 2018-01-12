s#Extract
# Read the query file
#
#
# from keras.applications.vgg16 import VGG16
# from keras.preprocessing import image
# from keras.applications.vgg16 import preprocess_input
import numpy as np
# from keras.preprocessing import image
import json
import skipthoughts
import pickle
import penseur

#modelSent= skipthoughts.load_model()


# model = VGG16(weights='imagenet', include_top=False)
with open('/home/subha/ReferralExp/ObjectRetreival/natural-language-object-retrieval/data/metadata/referit_query_dict.json') as fp:
    d = json.load(fp)

#print(d)
#1. list of sentences
#2. list of numpy arrays
#3. skip thought vectors
train=[]
dev=[]
sentences=[]
imgFeatures=[]
skipThought=[]

img_id=[]
caption=[]

for key, value in d.iteritems():
    img_id.append(key)
    caption.append(value)
print(len(img_id))
print(len(caption))
#print(caption)
for i in range(len(caption)):
    print(i)
    img = image.load_img('/home/subha/ReferralExp/ObjectRetreival/natural-language-object-retrieval/data/resized_imcrop/'+img_id[i]+'.png'
    , target_size=(224, 224))
    x=image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    features=model.predict(x)
    imgFeatures.append(features)
    #sentences.append(caption[i])
    #print(sentences):
    #textFeatures=skipthoughts.encode(modelSent,[caption[i]])
    #skipThought.append(textFeatures)
with open("imgFeatures.txt","wb") as fp:
    pickle.dump(imgFeatures,fp)
# print(sentences,imgFeatures,skipThought)

#print(img)
