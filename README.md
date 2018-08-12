# Review and Rating Prediction System
##### By Aakash Moghariya
##### Safetytek

This repository contains program to train a neural network model on customer review dataset using Keras. The trained network is then freezed and served as a tensorflow model using flask-restplus framework. 


#### Technology Used
 - Python 2.7
 - Keras (With tensorflow as backend)
 - Tensorflow 
 - Flask-restplus
 - HTML
 - JQuery
 - Pandas
 - Numpy
 

#### Deployment instruction
1. For training and saving both models nlp_model.py is used, example for running the code are illustrated below. Default value for file_path is 'data.csv', for tokenizer_name is 'tokenizer.pickle', for rating_epochs is 50, for recom_epochs is 50, for batch_size is 200, for rating_model is 'rating_model.h5' and for recom_model is 'recommendation_model.h5'. To understand more about how to run the program pelase use --help.
    ```Bash
    python nlp_model.py
    python nlp_model.py --file_path 'data.csv' --tokenizer_name 'tokenizer.pickle' --rating_epochs 50 --recom_epochs 50 --batch_size 200 --rating_model 'rating_model.h5' --recom_model 'recommendation_model.h5'
    ```
2. To freeze the trained model use freeze.py. Default for the saved keras model is rating_model.pb, frozen model default would be 'rating.pb' and default path for the saving frozen model is the current working directory. The program could be executed as follows
    ```Bash
    python freeze.py
    python freeze.py --model_name rating_model.h5 --frozen_model_name rating.pb --frozen_model_path /home/ubuntu
    ```
3. Copy the forzen models to ../restplus/api/model/model_file/
4. Copy the fiited enocders to the ../restplus/api/model/model/endpoints
5. To run rest api to serve tesnorflow model go to ../restplus/settings.py modify the host name and other settings as needed. Then modify the model arguments at ../restplus/api/model/modelconfig.py
    ```bash
    python restplus/app.py
    ```
6. Swagger UI is delpoyed at http://hostname:port. It can be used by developers to develop and test APIs by means of sending sample request to the API and receiving the response. 
7. Ensure corss-origin-request is turned on in your browser.
8. Modify the request url in the ajax request of index.html. Run index.html to test the app.
9. For getting reivew and rating prediction on a .csv dataset file, please use test.py file. Default value of the arguments are specified below, plesae use --help to know more about the arguments.
    ```Bash
    python test.py
    python test.py --file_path data.csv --tokenizer tokenizer.pickle --rating_model rating_model.h5 --recom_model recommendation_model.h5
    ```

**Note**: To view the model architecture of the trained model open rating_model.png and recommendation_model.
**Note**: To view ui photos please use ui.png and ui1.png

#### Pretrained Model Download and Encoder Download
To download a pretrained keras as well as frozen mode use the google drive link specified below. 
   
[Pre Trained Models and Encoders](https://drive.google.com/drive/folders/1452PSgaAkvA0jvQIl2TVLPK4dekirs4D?usp=sharing)