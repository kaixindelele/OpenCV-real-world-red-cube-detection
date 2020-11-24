# -*- encoding: utf-8 -*-
'''
@File    :  detection.py
@Time    :  2019/11/10 01:01:20
@Author  :  Li Xiang
@Version :  1.0
@Contact :  lixiang6632@outlook.com
@License :  (C)Copyright 2019-2020, Song-Group-IIM-R&A
@Describe:  this module is used to detect the cube and calculate the coordinates of cube
            under robot's base coordinate system
            functions:
                extract_red, extract_yellow, extract_gray, extract_background，find_two_points，
                calculate_centroid，calculate_distance，calculate_coordinate_of_robot
'''


from math import sqrt
# from cv2 import cv2 as cv
import cv2
import numpy as np
from sympy import symbols, solve


def extract_orange(img):
    """
        extract green object from the input image
        parameter: image
        return: green object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the first area
    lower_green=np.array([11,43,46])
    upper_green=np.array([25,255,255])

    mask0 = cv2.inRange(img_hsv, lower_green, upper_green)
    # splice two areas
    mask = mask0
    return mask


def extract_red(img, show_flag=False):
    """
        extract red object from the input image
        parameter: image
        return: red object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
    cv2.imshow('mask0', mask0)
    cv2.waitKey(0)
    # extract the second area
    # lower_red = np.array([156, 43, 46])
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
    cv2.imshow('mask1', mask1)
    cv2.waitKey(0)
    # splice two areas
    mask = mask0 + mask1
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    if show_flag:
        return img_hsv
    else:
        return mask

def extract_white(img):
    """
        extract green object from the input image
        parameter: image
        return: green object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the first area
    lower_green=np.array([0,0,0])
    upper_green=np.array([180,50,255])
    mask0 = cv2.inRange(img_hsv, lower_green, upper_green)
    # splice two areas
    mask = mask0
    cv2.imshow("black", mask0)
    cv2.waitKey(0)
    return mask

def extract_black(img):
    """
        extract green object from the input image
        parameter: image
        return: green object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the first area
    lower_green=np.array([0,0,0])
    upper_green=np.array([180,255,46])
    mask0 = cv2.inRange(img_hsv, lower_green, upper_green)
    # splice two areas
    mask = mask0
    cv2.imshow("black", mask0)
    cv2.waitKey(0)
    return mask

def extract_green(img):
    """
        extract green object from the input image
        parameter: image
        return: green object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the first area
    lower_green=np.array([35,43,46])
    upper_green=np.array([77,255,255])

    mask0 = cv2.inRange(img_hsv, lower_green, upper_green)
    # splice two areas
    mask = mask0
    return mask

def extract_cyan(img):
    """
        extract green object from the input image
        parameter: image
        return: green object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # extract the first area
    lower_green=np.array([78,100,46])
    upper_green=np.array([110,255,255])

    mask0 = cv2.inRange(img_hsv, lower_green, upper_green)
    # splice two areas
    mask = mask0
    return mask


def extract_yellow(img):
    """
        extract yellow object from the input image
        parameter: image
        return: yellow object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the yellow area
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
    return mask


def extract_gray(img):
    """
        extract gray object from the input image
        parameter: image
        return: gray object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the gray area
    lower_gray = np.array([0, 0, 46])
    upper_gray = np.array([180, 43, 220])
    mask = cv2.inRange(img_hsv, lower_gray, upper_gray)
    return mask

def extract_blue(img):
    """
        extract gray object from the input image
        parameter: image
        return: gray object of the image
    """
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # extract the gray area
    lower_gray = np.array([90, 43, 46])
    upper_gray = np.array([110, 255, 255])
    mask = cv2.inRange(img_hsv, lower_gray, upper_gray)
    return mask

def extract_background(img, background_points):
    """
        extract background from the input image
        parameters:
            image
            background_points: an array of points
        return: background of the image
    """
    four_points = np.array(background_points, dtype=np.int32)
    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, four_points, 1)
    mask = mask.astype(np.bool)

    res = np.zeros_like(img)
    res[mask] = img[mask]
    return res


def find_two_points(img):
    """
        obtain a pair of vertices of the cube from the object extracted image
        parameter: object extracted image
        return: a numpy array of two points
    """
    row_sum = list(img.sum(1))
    row_useful = [x for x in row_sum if x > 0]
    index_row_first = row_sum.index(row_useful[0])
    index_row_second = index_row_first + len(row_useful) - 1
    index_col_first = list(img[index_row_first]).index(255) + row_useful[0] // 255 // 2
    index_col_second = list(img[index_row_second]).index(255) + row_useful[-1] // 255 // 2
    return np.array([[index_row_first, index_col_first], [index_row_second, index_col_second]])


def calculate_centroid(point_a, point_b):
    """
        calculate the midpoint of the two input points
        parameter: two points of type: array
        return: midpoint of type: array
    """
    return [(point_a[i] + point_b[i]) / 2 for i in range(3)]


def calculate_distance(point_a, point_b):
    """
        calculate European distance of the two points in space
        parameter: two points of type: array
        return: European distance of tpye: float
    """
    dis = sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2 + \
         (point_a[2] - point_b[2]) ** 2)
    return dis


def calculate_coordinate_of_robot(point_a, radius_a, point_b, radius_b, delta_h, side_length):
    """
        calculate the intersection of two spheres, then find the position according to
        the height(delta_h + side_length / 2)
        parameters:
            point_a: center point of the first sphere
            radius_a: radius of the first sphere
            point_b: center point of the second sphere
            radius_b: radius of the second sphere
            delta_h: the difference between the heights of object desk and the robot desk
            side_length: side length of the cube
        return: the coordinate in robot's base coordinate system
    """
    # import math

    # if math.isnan(side_length):
    #     print("side_length:", side_length)
    #     side_length = 48

    coordinate_x, coordinate_y, coordinate_z = symbols('coordinate_x coordinate_y coordinate_z')
    res = solve([(coordinate_x - point_a[0]) ** 2 + (coordinate_y - point_a[1]) ** 2 + \
            (coordinate_z - point_a[2]) ** 2 - radius_a ** 2, (coordinate_x - point_b[0]) ** 2 + \
            (coordinate_y - point_b[1]) ** 2 + (coordinate_z - point_b[2]) ** 2 - radius_b ** 2, \
            coordinate_z - delta_h - side_length / 2], [coordinate_x, coordinate_y, coordinate_z])
    res = [coordinate_x for coordinate_x in res if coordinate_x[1] < -40]
    return list(res[0])


def test():
    """
        just for test
    """
    point_dest_a = [400, -400, 0]
    point_dest_b = [-400, -400, 0]
    radius_a, radius_b = 800, 800
    delta_h, side_length = 0.5, 4
    res = calculate_coordinate_of_robot(point_dest_a, radius_a, \
    point_dest_b, radius_b, delta_h, side_length)
    print(res)


if __name__ == "__main__":
    test()
