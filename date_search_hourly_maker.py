import numpy as np

date_file = open("date_search_2017.txt","r")
date_file_r = date_file.read()
dates = date_file_r.split("\n")

hourly_file = open("hourly_since_biden.txt","w")
for date in dates:
    hourly_file.write(date)
    hourly_file.write("\n")
    hourly_file.write(date[:11]+"08:00:00.000Z")
    hourly_file.write("\n")
    hourly_file.write(date[:11]+"16:00:00.000Z")
    hourly_file.write("\n")
