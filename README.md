# Image Captioning with Tensorflow and Keras and deployment using Flask and Docker in Heroku
![alt text](https://github.com/AhmedaliElgabry/Image_captioning/blob/master/model%20output%20example.png)
***This repository contains two parts:***<br/>
## 1)Train image captioning model- check the current directory - :<br/>
-the model trained in Flicker8k_Dataset which contains around 40 thousand image and their corresponding captions-you can download the dataset from here [here](https://machinelearningmastery.com/prepare-photo-caption-dataset-training-deep-learning-model/)-.<br/>
-have used glove embeddings and LSTM layer to capture text information.<br/>
-have uesd inceptionv3 pretrained model - it was trained in imagenet - to capture image information.<br/>
-the model trained in GCP for 3 epchos and the output model is saved-caption_model.model-.<br/>
-you can pass training process and use my trained model directly but be aware the if you can train the model for more epochs that will increase the performance a lot,actually i did`t do that because it will cost me a lot :"D. <br/>
			<br/>
## 2)Deploy this model into production using Flask , Docker and Heroku- check the deployment directory -<br/>
-you can visit demo here [https://imagecaptioningdemo.herokuapp.com/](https://imagecaptioningdemo.herokuapp.com/)

