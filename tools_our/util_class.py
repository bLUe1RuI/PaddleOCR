import cv2
import numpy as np
import math
from shapely import Polygon

class Sector:
    def __init__(self, points, angle):
        # self.points = points
        self.sector = Polygon(np.array(points).astype('int'))
        self.angle = angle
        if self.angle > 90:
            self.rotate_angle = 90 + 360 - self.angle
        else:
            self.rotate_angle = 90 - self.angle
        # first det
        self.with_det = False
        self.with_det_thre = None
        self.det_box = None
        self.is_core = False
        # second det and rec
        self.first_det_file = None
        self.ocr_boxes = None
        self._transcriptions_str = None
        
        
    def cal_contail_ocr(self, det_box, thre=0.6):
        box_contour = Polygon(det_box['points'])
        intersection = self.sector.intersection(box_contour)
        conf = intersection.area / box_contour.area
        if conf > thre:
            self.with_det = True
            self.det_box = det_box
            self.with_det_thre = conf

class CircleAixs:
    def __init__(self, bigcircle, secondcircles):
        self.center_thre = 15
        self.radius = bigcircle[1]
        self.centx, self.centy, post_circles = self.infer_center(bigcircle, secondcircles)
        self.axis_angle = self.infer_axis_angle(post_circles)
        
    def infer_center(self, bigcircle, secondcircles):
        post_circles = []
        center_circles = []
        for circle in secondcircles:
            x1, y1 = bigcircle[0]
            x2, y2 = circle[0]
            dis = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dis < self.center_thre:
                center_circles.append(circle)
            else:
                post_circles.append(circle)
        if len(center_circles) > 1:
            center_circle = sorted(center_circles, key=lambda x: -x[2])[0]
        elif len(center_circles) == 0:
            center_circle = bigcircle
        else:
            center_circle = center_circles[0]
        centx, centy = center_circle[0]
        return centx, centy, post_circles
    
    def modify_det_box(self, det_box_json):
        for det_box in det_box_json['boxes']:
            _boxes = det_box['points']
            new_boxes = []
            assert len(_boxes) == 4
            _boxes = sorted(_boxes, key=lambda x, centx=self.centx, centy=self.centy: -np.linalg.norm(np.array(x) - [centx, centy]))
            first_second_boxes = _boxes[:2]
            angle1 = int(math.degrees(math.atan2(-(first_second_boxes[0][1] - self.centy), first_second_boxes[0][0] - self.centx)))
            angle2 = int(math.degrees(math.atan2(-(first_second_boxes[1][1] - self.centy), first_second_boxes[1][0] - self.centx)))
            angle1 = self.set_360(angle1)
            angle2 = self.set_360(angle2)
            max_angle = max([angle1, angle2])
            min_angle = min([angle1, angle2])
            if max_angle - min_angle > 180:
                if angle1 == min_angle:
                    new_boxes.append(first_second_boxes[0])
                    new_boxes.append(first_second_boxes[1])
                else:
                    new_boxes.append(first_second_boxes[1])
                    new_boxes.append(first_second_boxes[0])
            else:
                if angle1 == max_angle:
                    new_boxes.append(first_second_boxes[0])
                    new_boxes.append(first_second_boxes[1])
                else:
                    new_boxes.append(first_second_boxes[1])
                    new_boxes.append(first_second_boxes[0])
            
            dis3 = np.linalg.norm(np.array(_boxes[2]) - new_boxes[-1])
            dis4 = np.linalg.norm(np.array(_boxes[3]) - new_boxes[-1])
            if dis3 <= dis4:
                new_boxes.append(_boxes[2])
                new_boxes.append(_boxes[3])
            else:
                new_boxes.append(_boxes[3])
                new_boxes.append(_boxes[2])
            det_box['points'] = new_boxes     
    
    def set_360(self, angle):
        while(angle < 0 or angle > 360):
            if angle < 0:
                angle += 360
            else:
                angle -= 360
        return angle
    
    def set_scale(self, scale):
        self.radius = int(self.radius * scale)
        self.centx = int(self.centx * scale)
        self.centy = int(self.centy * scale)
                
    def cal_points(self, start_angle, end_angle, num_segmets=5):
        angles = [(start_angle + i * (end_angle - start_angle) / num_segmets) for i in range(num_segmets + 1)]
        points = [(self.centx + self.radius * math.cos(math.radians(angle)), self.centy - self.radius * math.sin(math.radians(angle))) for angle in angles]
        return points
        
    def cal_sector(self):
        sectors = []
        for i in range(len(self.axis_angle)):
            if i == len(self.axis_angle) - 1:
                angle1 = self.axis_angle[i]
                angle2 = self.axis_angle[0] + 360
            else:
                angle1 = self.axis_angle[i]
                angle2 = self.axis_angle[i+1]
            sector_points = [(self.centx, self.centy)]
            sector_points.extend(self.cal_points(angle1, angle2))
            
            mid_angle = (angle1 + angle2) // 2
            sectors.append(Sector(sector_points, mid_angle))
        return sectors
    
    def infer_axis_angle(self, post_circles):
        circle_angle = []
        if len(post_circles) == 0:
            return circle_angle
        if len(post_circles) == 1:
            x, y = post_circles[0][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle = int(math.degrees(math.atan2(dy, dx)))
            circle_angle.append(self.set_360(angle))
            circle_angle.append(self.set_360(angle + 120))
            circle_angle.append(self.set_360(angle + 240))
        elif len(post_circles) == 2:
            x, y = post_circles[0][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle1 = int(math.degrees(math.atan2(dy, dx)))
            x, y = post_circles[1][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle2 = int(math.degrees(math.atan2(dy, dx)))
            angle1 = self.set_360(angle1)
            angle2 = self.set_360(angle2)
            circle_angle.append(angle1)
            circle_angle.append(angle2)
            
            min_angle = min([angle1, angle2])
            max_angle = max([angle1, angle2])
            if max_angle - min_angle > 180:
                angle3 = max_angle - (max_angle - min_angle) // 2
            else:
                angle3 = self.set_360(max_angle + 120)
            circle_angle.append(angle3)
        else:
            x, y = post_circles[0][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle1 = int(math.degrees(math.atan2(dy, dx)))
            
            x, y = post_circles[1][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle2 = int(math.degrees(math.atan2(dy, dx)))
            
            x, y = post_circles[2][0]
            dx = x - self.centx
            dy = -(y - self.centy)
            angle3 = int(math.degrees(math.atan2(dy, dx)))
            
            angle1 = self.set_360(angle1)
            angle2 = self.set_360(angle2)
            angle3 = self.set_360(angle3)
            circle_angle.append(angle1)
            circle_angle.append(angle2)
            circle_angle.append(angle3)
        circle_angle = sorted(circle_angle)
        
        return circle_angle   
            

class AxisDetect:
    def __init__(self, debug=False):
        self.debug = debug
    
    def detect_circle_houghcircles(self, image, gray, param1, param2, minRadius, maxRadius):
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=param1, param2=param2, 
                                minRadius=minRadius, maxRadius=maxRadius)
        circles = np.uint16(np.around(circles))
        
        if self.debug:
            for i in circles[0, :]:
                cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.imshow('image', image)
            cv2.waitKey(0)
            
        post_cnts = []
        for i in circles[0, :]:
            post_cnts.append(((i[0], i[1]), i[2], 5))
        return image
    
    def detect_circle_morphology(self, image, gray, thre_a=5, thre_area1=1000, thre_area2=10000):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=3)
        
        cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # print(len(cnts))
        post_cnts = []
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            area = cv2.contourArea(c)
            # print(f"approx: {len(approx)}, area: {area}")
            if len(approx) >= thre_a and area > thre_area1 and area <= thre_area2:
                ((x, y), r) = cv2.minEnclosingCircle(c)
                if self.debug:
                    cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), 3)
                post_cnts.append(((x, y), r, len(approx)))

        if self.debug:
            cv2.imshow('image', image)
            cv2.waitKey(0)
        return post_cnts
    
    def infer_circle(self, tgt_image):
        tgt_height, tgt_width, _ = tgt_image.shape
        # det first circle
        first_image = tgt_image.copy()
        first_image = cv2.medianBlur(first_image, 11)
        gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        bigcircles = self.detect_circle_morphology(tgt_image, gray, 3, 10000, 50000)
        if len(bigcircles) >= 1:
            bigcircle = sorted(bigcircles, key=lambda x: -x[1])[0]
        else:
            bigcircles = self.detect_circle_houghcircles(tgt_image, gray, 180, 40, 100, 150)
            if len(bigcircles) >= 1:
                bigcircle = sorted(bigcircles, key=lambda x: -x[1])[0]
            else:
                return None, None
            
        # second circle
        mask = np.zeros([tgt_height, tgt_width])
        (x, y), r, _ = bigcircle
        cv2.circle(mask, (int(x), int(y)), int(r), 1, -1)
        mask = mask.astype('uint8')
        mask_image = mask[:,:,None] * tgt_image
        mask_image = cv2.medianBlur(mask_image, 11)
        gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        gray = mask * gray
        secondcircles = self.detect_circle_morphology(first_image, gray, 6, 600, 1400)
        if len(secondcircles) > 4:
            secondcircles = sorted(secondcircles, key=lambda x: -x[2])[:4]
        return bigcircle, secondcircles
    
    def infer_axis(self, bigcircle, secondcircles):
        return CircleAixs(bigcircle, secondcircles)
    
    def infer(self, image):
        # preprocess
        height, width, _ = image.shape
        tgt_width = 512
        tgt_height = int(tgt_width / width * height)
        tgt_image = cv2.resize(image, (tgt_width, tgt_height))
        # detect circle
        bigcircle, secondcircles = self.infer_circle(tgt_image)
        if bigcircle is None:
            return None, "warning: 未检测到关键轴，请转动适当角度"
        circleaxis = self.infer_axis(bigcircle, secondcircles)
        if len(circleaxis.axis_angle) == 0:
            return None, "warning: 未检测到关键轴，请转动适当角度"
        circleaxis.set_scale(width / tgt_width)
        return circleaxis, "关键轴检测成功~~~"
        
        