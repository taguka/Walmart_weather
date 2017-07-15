import os
import pandas as pd
import numpy as np

dir_files='C:/Kaggle/Walmart_weather/output/'
list_files=os.listdir('C:/Kaggle/Walmart_weather/output')
submit_file='C:/Kaggle/Walmart_weather/output/submission.csv'
with open(submit_file, 'w') as outfile:
    outfile.write('id,units\n')
    for fname in list_files:
        with open(dir_files+fname) as infile:
            outfile.write(infile.read())
            