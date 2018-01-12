#edit
import pickle
import os
import numpy as np
import random
import json

train=[]
dev=[]
sent=[]
imgFeat=[]
sentFeat=[]
trainNeg=[]

def load_pickle(name):
    print(name)
    with open(name,"rb") as fp:
        l=pickle.load(fp)

    return l

textNames=os.listdir('textFeaturescache')
# print(textNames)
imgCache=[]
imgId=[]
X2Train=[]
#with open("img_id.txt","rb") as fp:
imgId=load_pickle("img_id.txt")
cap=load_pickle("captions.txt")
# imgF = load_pickle("imgFeatures.txt")

for i in textNames:
    # print(i)
    # print(textNames[i])
    # num= random.choice(range(i)+range(i+1,len(textNames)))
    # n=textNames[i]
    # output_file = open('/home/subha/ReferralExp/ObjectRetreival/natural-language-object-retrieval/data/metadata/referit_imcrop_bbox_dict.json').read()
    # output_json = json.loads(output_file)

    # print(n)
    sF=load_pickle('textFeaturescache/'+i)
    #
    # # print(sF)
    sF=np.array(sF).flatten()
    X2Train.append(sF)
    # print(sF.shape)
    # a=n.split('_',1)[1]
    # b=a.split('.')[0]
    # ind=imgId.index(b)
    # coord = output_json[b]
    # # print(b)
    # c=b.split('_')[0]
    # c=c+'_fc7.npy'
    # f=np.load('/home/subha/ReferralExp/ObjectRetreival/natural-language-object-retrieval/data/referit_context_features/'+c)
    # iF=np.concatenate((imgF[ind].flatten(),f.flatten()),axis=0)
    # iF = np.concatenate(imF,np.array(coord),axis=0)
    # FF=np.concatenate((iF, sF),axis=0)
    # print(FF.shape)
    # train.append(iF)
    # np.save('/home/subha/RefExpCVPR2018/penseur/finalTrain/imgCache/'+n+'.npy',iF)


X2Train=np.array(X2Train)
np.save('/home/subha/RefExpCVPR2018/penseur/finalTrain/X2Train.npy',X2Train)
# trainNegFinal=trainNeg[0:80000]
# testNegFinal=trainNeg[80001:]
# np.save('trainNegFinal.npy',trainNegFinal)
# np.save('testNegFinal.npy',testNegFinal)

# trainFinal=np.array(train[0:80000])
# testFinal=np.array(train[80001:])
# np.save('train.npy',trainFinal)
# np.save('test.npy',testFinal)

    # print(c)
#     imgFeat.append(imgF[ind].flatten())
#     sentFeat.append(np.array(sF).flatten())
#     for j in cap[ind]:
#         sent.append(j)
# #
# train.append(sent[1:350])
# train.append(imgFeat[1:350])
# train.append(sentFeat[1:350])
# dev.append(sent[351:499])
# dev.append(imgFeat[351:499])
# dev.append(sentFeat[351:499])
#
# with open("train.txt","wb") as fp:
#     pickle.dump(train,fp)
#
# with open("dev.txt","wb") as fp:
#     pickle.dump(dev,fp)
 # print(train[0],train[2])
