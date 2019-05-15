import string
from captcha.image import ImageCaptcha
import pandas as pd
import numpy as np

from PIL import ImageFont

### Image Setup
image = ImageCaptcha(fonts = [r"./Lato-Black.ttf",
                              r"./fonts/lato/Lato-Regular.ttf",
                              r"./fonts/lato/Lato-Bold.ttf",
                              r"./fonts/lato/Lato-Light.ttf",
                              r"./fonts/quicksand/Quicksand-Regular.otf",
                              r"./fonts/quicksand/Quicksand-Bold.otf",
                              r"./fonts/quicksand/Quicksand-Light.otf",
                              r"./fonts/open-sans/OpenSans-Regular.ttf",
                              r"./fonts/open-sans/OpenSans-Bold.ttf",
                              r"./fonts/open-sans/OpenSans-Light.ttf",
                              r"./fonts/raleway/Raleway-Bold.ttf",
                              r"./fonts/raleway/Raleway-Light.ttf",
                              r"./fonts/raleway/Raleway-Regular.ttf",
                              r"./fonts/vera/Vera.ttf",
                              r"./fonts/vera/VeraBd.ttf",
                              ], width = 32, height = 32)

# Create Folders
import os
import time

start_time = time.time()
print ("Code start ...")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

# Generate Data

times = 1000                                   # number of pictures we want to generate for a single character

letters_total = string.ascii_letters[:]

count = 1
save_count = []
save_char = []

createFolder('./images/')
number_dig = 0
#iterate for number image
for number in range(0, 10):
    for i in range (1, times+1):
        image.write(str(number),  './images/' + str(count) + '.png')
        save_count.append(count)
        save_char.append(number_dig)
        count = count + 1
    #+1
    number_dig+=1

print('Finished number')

#字母位数10
letter_dig = number_dig
for letter in letters_total:
    #第几个字母
    for i in range (1, times+1):
        image.write(str(letter), './images/' + str(count) + '.png')
        save_count.append(count)
        save_char.append(letter_dig)
        count = count + 1

    #+1
    letter_dig +=1

data = pd.DataFrame({"count" : save_count, "character" : save_char})

header = ["count", "character"]

print(data)
data.to_csv('data.csv', encoding = 'utf-8', columns = header)

end_time = time.time()

print ("Total run time for generating one character data (60x60) 100 times per character:" + str(end_time - start_time), "seconds in total")
