<h1> Age_Classification </h1>

The Age Classifier is a machine learning model designed to estimate the age and gender of individuals based on input features such as facial images. This project aims to provide a robust and accurate tool for age and gender estimation.
<br>
<br>
It is built using python and it's libraries such as tensorflow, streamlit, etc.
<br>
<h2>Features</h2>
<li>The data used to train the models is taken from UTKfacedataset which can be found on online.</li>
<li>The image data is preprocessed and trained on a cnn model.</li>
<li>This project uses two different models for age and gender each the only difference being the output layer and the metrics to evaluate the model accuracy.</li>
<li>The web-interface is built using streamlit which allows easy and quick building of interfaces using a variety of features available. The interface is simple and intuitive making it user-friendly.</li>
<li>The project can deployed via streamlit</li>
<br>
<h2>Usage</h2>
<li>To use the project locally, just clone the repository and install the requirements via "python -m pip install -r requirements.txt".</li>
<li>To use pretrained model, just run "streamlit run app.py" and you will be directed to hosted port on browser.</li>
<li>Also the models can be trained on custom data stored in "\notebook\data\".</li>
<br>
<strong> Note: </strong> Due to hardware limitations, currently the models are trained on less data of around 10k images but can be imporved further using more data for training and fine-tuning the model architecture to achieve better accuracy.

Some screenshots of the project are:-

![Screenshot (98)](https://github.com/OmSky1/Age_Classifier/assets/119601753/307fd020-d5e5-42c5-88f9-bbbef1b8ca38)
![Screenshot (99)](https://github.com/OmSky1/Age_Classifier/assets/119601753/e6397dad-b3ec-4408-88f6-3cfffbf04c5b)


