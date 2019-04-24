import cv2
import numpy as np
import matplotlib.pyplot as plt

def Projection_H(img, height, width):
    array_H = np.zeros(height)
    for i in range(height):
        total_count = 0
        for j in range(width):
            temp_pixVal = img[i, j]
            if (temp_pixVal == 0):
                total_count += 1
        array_H[i] = total_count

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_axis = np.arange(height)
    ax.barh(x_axis, array_H)
    fig.savefig("hist_H.png")

    return array_H


def Projection_V(img, height, width):
    array_V = np.zeros(width)
    for i in range(width):
        total_count = 0
        for j in range(height):
            temp_pixVal = img[j, i]
            if (temp_pixVal == 0):
                total_count += 1
        array_V[i] = total_count


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    x_axis = np.arange(width)
    ax.bar(x_axis, array_V)
    fig.savefig("hist_V.png")

    return array_V


def Detect_HeightPosition(H_THRESH, height, array_H):
    lower_posi = 0
    upper_posi = 0

    for i in range(height):
        val = array_H[i]
        if (val > H_THRESH):
            lower_posi = i
            break

    for i in reversed(range(height)):
        val = array_H[i]
        if (val > H_THRESH):
            upper_posi = i
            break

    return lower_posi, upper_posi


def Detect_WidthPosition(W_THRESH, width, array_V):
    char_List = np.array([])

    flg = False
    posi1 = 0
    posi2 = 0
    for i in range(width):
        val = array_V[i]
        if (flg==False and val > W_THRESH):
            flg = True
            posi1 = i

        if (flg == True and val < W_THRESH):
            flg = False
            posi2 = i
            char_List = np.append(char_List, posi1)
            char_List = np.append(char_List, posi2)

    return char_List


if __name__ == "__main__":
    # input image
    img = cv2.imread("./character.jpg")

    # convert gray scale image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # black white
    ret, bw_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU)
    height, width = bw_img.shape

    # create projection distribution
    array_H = Projection_H(bw_img, height, width)
    array_V = Projection_V(bw_img, height, width)

    # detect character height position
    H_THRESH = 5
    lower_posi, upper_posi = Detect_HeightPosition(H_THRESH, height, array_H)

    # detect character width position
    W_THRESH = 2
    char_List = Detect_WidthPosition(W_THRESH, width, array_V)

    # draw image
    if (len(char_List) % 2) == 0:
        print("Succeeded in character detection")
        for i in range(0, (len(char_List)-1), 2):
            img = cv2.rectangle(img, (int(char_List[i]), int(upper_posi)), (int(char_List[i+1]), int(lower_posi)), (0,0,255), 2)
        cv2.imwrite("result.jpg", img)
    else:
        print("Failed to detect characters")
