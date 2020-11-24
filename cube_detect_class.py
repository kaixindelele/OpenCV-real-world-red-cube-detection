import os
import math
import numpy as np
import cv2
import time
from detection import extract_red, extract_yellow, extract_gray, extract_orange, extract_black, extract_white
import pygame
from pygame.locals import QUIT


class Cube:
    def __init__(self,
                 input_image,                  
                 color='red',
                 show_flag=False,
                 points_num=6,
                 ):
        self.origin_img = input_image.copy()
        self.color = color
        self.points_num = points_num
        self.show_flag = show_flag
        self.surface_points_init
        canny_image = self.img_process_rgb2canny(input_image)
        self.canny_image = canny_image.copy()
        max_contour = self.img_process_canny2max_contour(canny_image)
        dark_with_poly = self.approx_polygone(max_contour)
        poly_6_points = self.get_poly_6_points(dark_with_poly,
                                               points_num=self.points_num,
                                               minDistance=50)        
        self.surface_points_update(poly_6_points)

        if self.show_flag:
            print("poly_6_points:", poly_6_points)   
            dark = self.origin_img.copy()
            for p in self.top_surface:
                cv2.circle(dark, (p[0], p[1]), 2, (200,0,0),-1)
            for p in self.buttom_surface:
                cv2.circle(dark, (p[0], p[1]), 2, (200,0,0),-1)
            cv2.imshow('poly_6_points', dark)
            cv2.waitKey(0)             
        
        cut_rate = 0.1
        new_six_lines_list = self.divide_6_lines_and_contours(poly_6_points, cut_rate)
        cross_6_points = self.update_6_corners_from_new_6_line_list(new_six_lines_list)
        if self.show_flag:
            print("cross_6_points:", cross_6_points)
            # dark = self.origin_img.copy()
            for p in cross_6_points:
                cv2.circle(dark, (p[0], p[1]), 2, (200,200,0),-1)
            self.surface_points_update(cross_6_points)
            for p in self.top_surface:
                cv2.circle(dark, (p[0], p[1]), 2, (200,200,0),-1)
            cv2.imshow('cross_6_points', dark)
            cv2.waitKey(0)       
        # self.surface_points_update(cross_6_points)


    def draw_two_points_line(self, input_img, point1, point2):                        
        # 起点和终点的坐标
        img = input_img.copy()
        ptStart = (point1[0], point1[1])
        ptEnd = (point2[0], point2[1])
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        cv2.imshow('image', img)
        cv2.waitKey(0)        

    def draw_one_point_line(self, input_img, point, line_ab):
        img = input_img.copy()
        k,b = line_ab[0], line_ab[1]
        x0, y0 = point[0], point[1]
        img = np.zeros((320, 320, 3), np.uint8) #生成一个空灰度图像
        # 起点和终点的坐标
        ptStart = (x0, y0)

        ptEnd = (x0+10, int(k*(x0+10)+b))
        point_color = (0, 255, 0) # BGR
        thickness = 1 
        lineType = 4
        cv2.line(img, ptStart, ptEnd, point_color, thickness, lineType)

        cv2.imshow('image', img)
        cv2.waitKey(0)        

    def calculate_dist_point2line(self, point, line_ab):
        k, b = line_ab[0], line_ab[1]
        x0, y0 = point[0], point[1]
        dist = (k*x0-y0+b)/np.sqrt(k*k+1)
        return dist

    def cross_point(self, line1_ab, line2_ab):#计算交点函数
        k1, b1 = line1_ab[0], line1_ab[1]
        k2, b2 = line2_ab[0], line2_ab[1]        
        x=(b2-b1)*1.0/(k1-k2)
        y=k1*x*1.0+b1*1.0
        return [int(x), int(y)]

    def calculate_from_two_points_to_line(self, point1, point2):
        x_list = [point1[0], point2[0]]
        y_list = [point1[1], point2[1]]
        k,b = self.Least_squares(x_list, y_list)
        return k, b

    def calculate_from_points_list_to_line(self, points_list):
        x_list = [p[0] for p in points_list]
        y_list = [p[1] for p in points_list]
        k,b = self.Least_squares(x_list, y_list)
        return k, b

    def Least_squares(self, x, y):        
        x_ = np.array(x).mean()
        y_ = np.array(y).mean()
        m = np.zeros(1)
        n = np.zeros(1)
        k = np.zeros(1)
        p = np.zeros(1)
        for i in np.arange(len(x)):
            k = (x[i]-x_)* (y[i]-y_)
            m += k
            p = np.square( x[i]-x_ )
            n = n + p
        a = m/n
        b = y_ - a* x_
        return a[0],b[0]

    def img_process_rgb2canny(self, input_image):        
        img = input_image
        if self.show_flag:
            cv2.imshow("img", img)
            cv2.waitKey(0)

        gray_img = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        gray_img = cv2.Canny(gray_img, 50, 150, apertureSize = 3)    
        if self.show_flag:
            cv2.imshow("gray_img", gray_img)
            cv2.waitKey(0)
            
        origin_img = img.copy()        
        red = extract_red(img)
        if self.show_flag:
            cv2.imshow("red", red)
            cv2.waitKey(0)
        # 去除一些噪点
        red = cv2.medianBlur(red, 5)    
        if self.show_flag:
            cv2.imshow("medianBlur", red)
            cv2.waitKey(0)
        gf_time = time.time()
        red = cv2.Canny(red, 50, 150, apertureSize = 3)    
        if self.show_flag:
            cv2.imshow("Canny", red)
            cv2.waitKey(0)
            
        return red
        
    def img_process_canny2max_contour(self, canny_image):
        contours, hierarchy = cv2.findContours(canny_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        
        max_cont = max([len(c) for c in contours])
        contours_corners = [c for c in contours if len(c)==max_cont][0]        
        contours_corners_array = np.array(contours_corners)
        corners = contours_corners_array    
        corners = np.int0(corners)
        max_contour = corners.reshape(corners.shape[0], 2)
        if self.show_flag:
            dark = self.origin_img.copy()
            for i in corners:
                x,y = i.ravel()
                cv2.circle(dark, (x, y), 1, (0, 200, 200), -1)                                
            cv2.imshow("max_contour", dark)
            cv2.waitKey(0)

            points_num=6
            level=0.1
            minDistance=50
            corners = cv2.goodFeaturesToTrack(canny_image,
                                            points_num,
                                            level,
                                            minDistance,
                                            )
            corners = np.int0(corners)
            corners = corners.reshape(corners.shape[0], 2)
            six_corners = corners[:, :]            
            dark = self.origin_img.copy()
            for i in corners:
                x,y = i.ravel()
                cv2.circle(dark, (x, y), 3, (0, 200, 200), -1)                                
            cv2.imshow("canny contours features", dark)
            cv2.waitKey(0)

        return max_contour

    def approx_polygone(self, max_contour,
                        points_num=6,
                        level=0.1,
                        minDistance=50):
        # approx polygone from contours!
        polygonelist = []
        corners = max_contour
        perimeter = cv2.arcLength(corners, True)
        epsilon = 0.005*cv2.arcLength(corners, True)
        approx = cv2.approxPolyDP(corners, epsilon, True)    

        polygonelist.append(approx)
        dark = np.zeros(self.origin_img.shape)
        cv2.drawContours(dark, polygonelist, -1, (255, 255, 255), 1)
        if self.show_flag:                      
            cv2.imshow("approx_corners", dark)
            cv2.waitKey(0)            
        dark_with_poly = dark.copy()
        return dark_with_poly

    def get_poly_6_points(self, dark_with_poly,
                          points_num=6,
                          minDistance=50):
        # poly features
        dark = dark_with_poly.astype('uint8')
        dark = cv2.cvtColor(dark, cv2.COLOR_RGB2GRAY)        
        # get 6 points!!!!
        level=0.1        
        corners = cv2.goodFeaturesToTrack(dark,
                                          points_num,
                                          level,
                                          minDistance,
                                          )
        corners = np.int0(corners)
        corners = corners.reshape(corners.shape[0], 2)
        six_corners = corners[:, :]
        if self.show_flag:
            dark = self.origin_img.copy()
            for i in corners:
                x,y = i.ravel()
                cv2.circle(dark, (x, y), 3, (0, 200, 200), -1)                                
            cv2.imshow("poly contours features", dark)
            cv2.waitKey(0)
        return six_corners

    def surface_points_update(self, poly_6_points):
        corners = poly_6_points
        data = corners[corners[:,1].argsort()]
        self.top_upper_point = data[0,:]        
        if data[1,0] < data[2,0]:
            self.top_left_point = data[1,:]
            self.top_right_point = data[2,:]
        else:
            self.top_left_point = data[2,:]
            self.top_right_point = data[1,:]        
        self.top_down_point = self.top_left_point + self.top_right_point - self.top_upper_point
        # 算出上表面中心点，虽然有畸变，但是应该够用了
        self.top_center_point = ((self.top_left_point + self.top_right_point)/2).astype(np.int16)

        # buttom surface points update
        data = corners[corners[:,1].argsort()] 
        self.buttom_down_point = data[-1,:]   
        # 哪个小，哪个是左        
        if data[-3,0] < data[-2,0]:
            self.buttom_left_point = data[-3,:]
            self.buttom_right_point = data[-2,:]
        else:
            self.buttom_left_point = data[-2,:]
            self.buttom_right_point = data[-3,:]     
        self.buttom_upper_point = self.buttom_left_point + self.buttom_right_point - self.buttom_down_point
        # 算出上表面中心点，虽然有畸变，但是应该够用了
        self.buttom_center_point = ((self.buttom_left_point + self.buttom_right_point)/2).astype(np.int16)
        
    def divide_6_lines_and_contours(self, poly_6_points, cut_rate = 0.1):
        other_conners = np.where(self.canny_image>0)
        other_conners = np.array([other_conners[1], other_conners[0]])         
        other_conners = other_conners.transpose()        
        # print("other_corners:", other_conners)
        corners = other_conners
        top_left_points_list = []
        top_right_points_list = []     
        mid_left_points_list = []
        mid_right_points_list = []     
        buttom_left_points_list = []
        buttom_right_points_list = []     
        # y_sorted_data = corners[corners[:,1].argsort()]
        y_sorted_data = corners
        
        for p in y_sorted_data:           
            if p[0] < self.top_upper_point[0] and p[1] < self.top_left_point[1]:
                top_left_points_list.append(p)
            elif p[0] > self.top_upper_point[0] and p[1] < self.top_right_point[1]:
                top_right_points_list.append(p)
            
            elif p[0] < self.top_upper_point[0] and p[1] > self.top_left_point[1] and p[1] < self.buttom_left_point[1]:
                mid_left_points_list.append(p)
            elif p[0] > self.top_upper_point[0] and p[1] > self.top_right_point[1] and p[1] < self.buttom_right_point[1]:
                mid_right_points_list.append(p)

            elif p[0] < self.buttom_down_point[0] and p[1] > self.buttom_left_point[1] and p[1] < self.buttom_down_point[1]:
                buttom_left_points_list.append(p)
            elif p[0] > self.buttom_down_point[0] and p[1] > self.buttom_right_point[1] and p[1] < self.buttom_down_point[1]:
                buttom_right_points_list.append(p)
                    
        if self.show_flag:        
            dark = self.origin_img.copy()
            for p in top_left_points_list:
                cv2.circle(dark, (p[0], p[1]), 1, (200,0,0),-1)
            cv2.imshow('left', dark)
            cv2.waitKey(0)
            dark = self.origin_img.copy()
            for p in top_right_points_list:
                cv2.circle(dark, (p[0], p[1]), 2, (200,100,0),-1)
            cv2.imshow('right_list', dark)
            cv2.waitKey(0)
            
            dark = self.origin_img.copy()
            for p in mid_left_points_list:
                cv2.circle(dark, (p[0], p[1]), 2, (200,100,0),-1)
            cv2.imshow('mid_left', dark)
            cv2.waitKey(0)
            dark = self.origin_img.copy()
            for p in mid_right_points_list:
                cv2.circle(dark, (p[0], p[1]), 2, (200,100,0),-1)
            cv2.imshow('mid_right', dark)
            cv2.waitKey(0)
            
            dark = self.origin_img.copy()
            for p in buttom_left_points_list:
                cv2.circle(dark, (p[0], p[1]), 2, (200,100,0),-1)
            cv2.imshow('buttom_left', dark)
            cv2.waitKey(0)
            dark = self.origin_img.copy()
            for p in buttom_right_points_list:
                cv2.circle(dark, (p[0], p[1]), 2, (200,100,0),-1)
            cv2.imshow('buttom_right', dark)
            cv2.waitKey(0)                    
        
        six_lines_list = [top_left_points_list,
                          top_right_points_list,
                          mid_left_points_list,
                          mid_right_points_list,
                          buttom_left_points_list,
                          buttom_right_points_list,]
        tem_six_lines_list = []
        for line_list in six_lines_list:
            points_num = len(line_list)
                        
            tem_line_list = line_list[int(cut_rate*points_num):-int(cut_rate*points_num)]
            tem_six_lines_list.append(tem_line_list)
            if self.show_flag:
                dark = self.origin_img.copy()
                for p in tem_line_list:
                    cv2.circle(dark, (p[0], p[1]), 1, (200,0,0),-1)
                cv2.imshow('cut_line', dark)
                cv2.waitKey(0)
                
        return tem_six_lines_list

    def update_6_corners_from_new_6_line_list(self, new_6_lines):        
        six_corners = []
        lines_kb_list = []
        for points_list in new_6_lines:
            k, b = self.calculate_from_points_list_to_line(points_list)
            lines_kb_list.append([k, b])
        six_corners.append(self.cross_point(lines_kb_list[0], lines_kb_list[1]))
        six_corners.append(self.cross_point(lines_kb_list[0], lines_kb_list[2]))
        six_corners.append(self.cross_point(lines_kb_list[1], lines_kb_list[3]))
        six_corners.append(self.cross_point(lines_kb_list[2], lines_kb_list[4]))
        six_corners.append(self.cross_point(lines_kb_list[3], lines_kb_list[5]))
        six_corners.append(self.cross_point(lines_kb_list[4], lines_kb_list[5]))
        six_corners = np.array(six_corners)
        return six_corners

    def surface_points_init(self,):
        self.top_upper_point = []
        self.top_down_point = []
        self.top_left_point = []
        self.top_right_point = []
        self.top_center_point = []
        self.buttom_upper_point = []
        self.buttom_down_point = []
        self.buttom_left_point = []
        self.buttom_right_point = []
        self.buttom_center_point = []
    @property
    def top_surface(self,):
        return [self.top_upper_point,
                self.top_down_point,
                self.top_left_point,
                self.top_right_point,
                self.top_center_point,]    
    @property
    def buttom_surface(self,):
        return [self.buttom_upper_point,
                self.buttom_down_point,
                self.buttom_left_point,
                self.buttom_right_point,
                self.buttom_center_point,]    



def main():    
    img = cv2.imread("input_image.jpg")
    st = time.time()
    cube_class = Cube(img, show_flag=True)
    print("st:", time.time()-st)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()