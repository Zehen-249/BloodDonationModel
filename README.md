# BloodDonationModel
This is logistic Regression Classification model to predict if a person with certain features will donate blood or not.

I found the training set data from [Data World](https://data.world/uci/blood-transfusion-service-center).

### Source of Data
Original Owner and DonorProf. I-Cheng YehDepartment of Information Management Chung-Hua University, Hsin Chu, Taiwan 30067, R.O.C.

## Training Set Data
uci-blood-*  contains the training set data. Required inforation for the dataset are writen in the **transfussion.names.txt** file.

## Files
- ### loadTraingData.py
    Load the data from csv file to numpy arrays
- ### Util.py
    This python file contains all the funtions for regularised Logistic Regression Model. 
    Futher explaination is given in the funtion body as well.

- ### trainModel.py
    Model is trained and the parameters are saved in a CSV File (param.csv).

- ### predict.py
    The model is used to make predictions on new data.