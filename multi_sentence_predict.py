import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

image = cv2.imread('images\Sentences\sen_6_copy.jpg')
image_copy = image.copy()

model = load_model('model_hand2.h5')
print('Model loaded.\n\n')

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}


gray = cv2.cvtColor (image_copy, cv2.COLOR_BGR2GRAY)

#brightness parameter
brightness = int(gray.mean())
print(brightness)

gray = cv2.GaussianBlur(gray, (7,7),500)


_, binary = cv2.threshold (gray, brightness-15, 255, cv2.THRESH_BINARY_INV)

custom_color_image = np.zeros_like(image_copy, dtype=np.uint8)

custom_color_image[binary == 255] = [0, 0, 0] 
custom_color_image[binary == 0] = [255, 255, 255]



plt.imshow(custom_color_image)
plt.show()
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_dist = 30

min_dist_x = 50

# Sort contours by y-values of the bounding rectangle
contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

for c in contours:
    print(cv2.boundingRect(c))

# Group contours based on y-values with a minimum distance for x-values
sorted_contours = []
current_group = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if not current_group:
        if w>20 and h>20:
            current_group.append((contour, x))
    else:
        prev_contour, prev_x = current_group[-1]
        if abs(y - cv2.boundingRect(prev_contour)[1]) <= min_dist:
            if w>20 and h>20 and w<200 and h<200:
                current_group.append((contour, x))
        else:
            sorted_group = sorted(current_group, key=lambda c: c[1])
            
            if w>20 and h>20:
                current_group = [(contour, x)]

                sorted_group_word = []
                sorted_group_word_list = [cv2.boundingRect(sorted_group[0][0])]

                for i in range (1, len(sorted_group)):
                    prev_contour, prev_x = sorted_group[i-1]
                    _, _, prev_w, _ = cv2.boundingRect(prev_contour)
                    x, y, w, h = cv2.boundingRect(sorted_group[i][0])

                    if abs(prev_x + prev_w - x)<=min_dist_x:
                        sorted_group_word_list.append(cv2.boundingRect(sorted_group[i][0]))
                    else:
                        sorted_group_word.append(sorted_group_word_list)
                        sorted_group_word_list= [cv2.boundingRect(sorted_group[i][0])]
                    
                sorted_group_word.append(sorted_group_word_list)

                sorted_contours.append(sorted_group_word)

# Append the last group of contours
if current_group:
    sorted_group = sorted(current_group, key=lambda c: c[1])


    sorted_group_word = []
    sorted_group_word_list = [cv2.boundingRect(sorted_group[0][0])]

    for i in range (1, len(sorted_group)):
        prev_contour, prev_x = sorted_group[i-1]
        _, _, prev_w, _ = cv2.boundingRect(prev_contour)
        x, y, w, h = cv2.boundingRect(sorted_group[i][0])

        if abs(prev_x + prev_w - x)<=min_dist_x:
            sorted_group_word_list.append(cv2.boundingRect(sorted_group[i][0]))
        else:
            sorted_group_word.append(sorted_group_word_list)
            sorted_group_word_list = [cv2.boundingRect(sorted_group[i][0])]
        
    sorted_group_word.append(sorted_group_word_list)

    sorted_contours.append(sorted_group_word)

sentence_list = []
line_list = []
word_list = []

for line in sorted_contours:
    for word in line:
        for letter_cont_rect in word:
            ind_word = custom_color_image[letter_cont_rect[1]:(letter_cont_rect[1]+letter_cont_rect[3]), letter_cont_rect[0]:letter_cont_rect[0]+letter_cont_rect[2]]
           
            #plt.imshow(ind_word)
            #plt.show()
            word_list.append(ind_word)
        line_list.append (word_list)
        word_list = []
    sentence_list.append(line_list)
    line_list = []

width = 50
height = 50

#prediction list
word_pred = []
line_pred = []
sentence_pred = []

word_group_list = []
sentence_group_list = []

for line in range(len(sentence_list)):
    for word in range(len(sentence_list[line])):
        for letter in range(len(sentence_list[line][word])):
            sentence_list[line][word][letter] = cv2.resize(sentence_list[line][word][letter], (width, height))

            sentence_list[line][word][letter] = cv2.copyMakeBorder(sentence_list[line][word][letter], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255,255,255))

            #plt.imshow(sentence_list[line][word][letter])
            #plt.show()
            
        indiv_word = cv2.hconcat( sentence_list[line][word])

        if word != len(sentence_list[line])-1:
            indiv_word = cv2.copyMakeBorder(indiv_word,0, 0, 0, 40, cv2.BORDER_CONSTANT, value=(255,255,255) )

        #plt.imshow(indiv_word)
        #plt.show()
        

        word_group_list.append (indiv_word)
    
    indiv_sentence = cv2.hconcat(word_group_list)
    sentence_group_list.append(indiv_sentence)
    word_group_list = []
    

for sentence in range(len(sentence_group_list)):
    sentence_group_list[sentence] = cv2.resize(sentence_group_list[sentence], (600, 50))
    
    if (sentence != len(sentence_group_list)-1):
        sentence_group_list[sentence] = cv2.copyMakeBorder(sentence_group_list[sentence],0,0,0,0,cv2.BORDER_CONSTANT, value=(255,255,255))

    plt.imshow(sentence_group_list[sentence])
    plt.show()

para = cv2.vconcat(sentence_group_list)


plt.imshow(para)
plt.show()


#prediction of the letters
for line in range(len(sentence_list)):
    for word in range(len(sentence_list[line])):
        for letter in range(len(sentence_list[line][word])):
        
            sentence_list[line][word][letter] = cv2.copyMakeBorder(sentence_list[line][word][letter] , 10,10,10,10,cv2.BORDER_CONSTANT, value=(255,255,255))
            sentence_list[line][word][letter] = cv2.cvtColor(sentence_list[line][word][letter], cv2.COLOR_BGR2GRAY)


            _, img_thresh = cv2.threshold(sentence_list[line][word][letter] , 150, 255, cv2.THRESH_BINARY_INV)

            img_final = cv2.resize(img_thresh, (28, 28))

        
            print(img_final.shape)

            img_final = np.reshape(img_final, (1, 28, 28, 1))

            

            img_pred = word_dict[np.argmax(model.predict(img_final))]
            

            word_pred.append(img_pred)
        line_pred.append(word_pred)
        word_pred = []
    sentence_pred.append(line_pred)
    line_pred = []

for line in sentence_pred:
    for word in line:
        for letter in word:
            print(letter,end='')
        print(' ', end='')
    print('\n', end = '')
    

        