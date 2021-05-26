import numpy as np
import cv2
import random
try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('\n[INFO]: Launching Tensorflow...')
    import tensorflow as tf
except:
    pass


drawing = False  # true if mouse is pressed
ix, iy = -1, -1  # ! these will be used to draw on the canvas
NUM1, NUM2 = -1, -1  # ! these are the numbers that will be shown in the math problem
PROBLEM_IMG_HEIGHT = 200
PROBLEM_IMG_WIDTH = 512
DRAWING_COLOR = (255, 255, 255)  # ! white

print('[INFO]: Loading The Model...')
MODEL = tf.keras.models.load_model('./english_digits_model')


def create_problem(change_the_problem=True):
    """
    this function will create a random math problem 
    also will create a drawing canvas so the user can draw the answer

    Args:
        change_the_problem (bool, optional): if the user didn't answer correctly.
        then we won't change the problem and we will make the problem font color RED
        Defaults to True.

    Returns:
        the problem image and the canvas on top of each other
    """

    global NUM1, NUM2
    if change_the_problem:
        NUM1 = random.randint(0, 9)
        NUM2 = random.randint(0, 9)

        #! make sure the summation is less than 10 so that the user will draw a single digit
        if NUM1 + NUM2 > 9:
            NUM1 = abs(6-NUM1)
            NUM2 = abs(5-NUM2)

    h, w = PROBLEM_IMG_HEIGHT, PROBLEM_IMG_WIDTH
    background = np.zeros((h, w, 3), np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (w//2-150, h//2)
    fontScale = 2
    fontColor = (255, 178, 150) if change_the_problem else (0, 0, 255)
    lineType = 2
    margin = 30

    cv2.putText(background, f"{NUM1} + {NUM2} = ?",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.line(background, (0, h-margin), (w, h-margin), fontColor, 10)
    return np.vstack((background,
                      np.zeros((400, 512, 3), np.uint8)))


def recognize_digit(img):
    """
    given the whole image crop the canvas part of the image resize it 
    then return the predicted number 

    Args:
        img (np.array): the problem+canvas image  

    Returns:
        int: the predcited number
    """
    global MODEL

    digit = cv2.resize(img[PROBLEM_IMG_HEIGHT:, :], (28, 28),)
    digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    digit = digit.reshape(1, 28, 28, 1).astype('float32') / 255

    return MODEL.predict(digit).argmax()


def paint_brush(event, x, y, flags, param):
    """
    this function will be always called to listen the movement of the mouse
    to react as a paint brush 
    """
    global ix, iy, drawing, img, DRAWING_COLOR, img
    if y < 190:
        return
    #! start drawing
    elif event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    #! keep drawing as the mouse is moving and mouse left button is pressed
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img, (x, y), 15, DRAWING_COLOR, -1)

    #! the mouse left button is released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img, (x, y), 15, DRAWING_COLOR, -1)

        #! recognize the number
        change_problem = False
        if NUM1 + NUM2 == recognize_digit(img):
            change_problem = True

        #! create a new problem
        img = create_problem(change_problem)


cv2.namedWindow('image')
cv2.setMouseCallback('image', paint_brush)
img = create_problem()

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
