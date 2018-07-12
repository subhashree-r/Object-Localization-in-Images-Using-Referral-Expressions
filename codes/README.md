

The Image and text features were cached as numpy which was later used for training. These are not attached due to the size limit.
```
finalImageFeatureExtrScript.py -This script reads the referitcoco dataset and extract the image features using pre-trained VGG from the
bounding boxes present within each image
-> finalImageFeatureConcat.py -This script cocatenates the bounding box image features with context features and the corresponding coordinates
->finalTextConcat.py - This script creates the training data for language features after extraction
-> finaltextFeatExtglove.py - This script extracts glove features of the queries and saves them
->finaltextFeatureExtkipth.py - This script extracts skip-thought vectors using penseur module
-> finaltextFeatureExtw2vec.py - This script extracts word2vec features using genism model and caches it
-> finalTestingScript.py- This script computes the recall of a model. Given a query it computes the distance with image features of all bounding boxes
within that image and retrieves that box with least distance.
-> finalTrainingScriptL2.py - Trains a MLP by reducing L2 loss between image and query features
->finalTrainingSiameseScript.py -Trains a MLP by reducing contrastive loss between image and query features
-> /notebooks/visEg1, visEg2 - Notebooks to visualize two example success cases
```
