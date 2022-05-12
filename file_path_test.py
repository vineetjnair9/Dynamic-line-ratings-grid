a = 1

import pickle
months = ['Jul']
str1 = "C:\\Users\\vinee\\OneDrive - Massachusetts Institute of Technology\\MIT\\Semesters\\Spring 2022\\15.S08\\Project\\weather_data_"
str2 = "2016.pkl"
file_name = str1 + months[0] + str2

with open(file_name, 'wb') as file:
    pickle.dump(a, file)