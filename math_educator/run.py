import numpy as np
import cv2
import os
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

#! notice that the dimensions below is for the math problem image that would be put
#! on top of the drawing canvas to form the final full image that would be shown to the user
PROBLEM_IMG_HEIGHT = 200  # ! height of the image problem
PROBLEM_IMG_WIDTH = 512  # ! width of the image problem

#! this is the paint-brush color
DRAWING_COLOR = (255, 255, 255)  # ! white

print('[INFO]: Loading The Model...')
# ! folder containing the current script
base_folder = os.path.dirname(__file__)
#! the full path of the saved model so that we can load it
path_model = os.path.join(base_folder, 'english_digits_model')
MODEL = tf.keras.models.load_model(path_model)


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
    #! generate random numbers
    if change_the_problem:
        NUM1 = random.randint(0, 9)
        NUM2 = random.randint(0, 9)

        #! make sure the summation is less than 10 so that the user will draw a single digit
        if NUM1 + NUM2 > 9:
            NUM1 = abs(6-NUM1)
            NUM2 = abs(5-NUM2)

    h, w = PROBLEM_IMG_HEIGHT, PROBLEM_IMG_WIDTH
    background = np.zeros((h, w, 3), np.uint8)  # ! black image

    #! write the math problem on the black background image
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (w//2-150, h//2)
    fontScale = 2
    fontColor = (0, 255, 0) if change_the_problem else (0, 0, 255)
    lineType = 2
    margin = 30

    cv2.putText(background, f"{NUM1} + {NUM2} = ?",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    #! but a line under the math problem to give the user a hint
    #!  about where the drawing canvas ends
    cv2.line(background, (0, h-margin), (w, h-margin), fontColor, 10)

    #! but the problem image on top of the drawing canvas
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
    #! resize the canvas image to (28x28)
    #! which is the accepted dimensions of the deep learning model
    digit = cv2.resize(img[PROBLEM_IMG_HEIGHT:, :], (28, 28))
    #! the model input image should be grayscale
    digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    #! normalize the input image
    digit = digit.reshape(1, 28, 28, 1).astype('float32') / 255

    return MODEL.predict(digit).argmax()  # ! the predicted number


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


window_name = 'image'
#! create a named widow so that we can track mouse movements inside it
cv2.namedWindow(window_name)
#! track the mouse movement
cv2.setMouseCallback('image', paint_brush)
img = create_problem()  # ! the first problem to show for the user
print('[INFO]: Running...')

while True:
    cv2.imshow('image', img)
    #! we need to call the line below even we don't use it anywhere
    #! but we need it so that we can show the image to the user
    keyCode = cv2.waitKey(1)

    #! exit the program be window exit button if we exit the window
    #! then its cv2.WND_PROP_VISIBLE would be 0.0
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        print('[INFO]: Exiting The Program...')
        cv2.destroyAllWindows()  # ! close any open windows
        break
