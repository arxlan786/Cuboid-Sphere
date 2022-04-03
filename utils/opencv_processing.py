import cv2
import math 
import numpy as np
import matplotlib.pyplot as plt


def show_image(image):
    cv2.imshow('image',image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def display_image(image_file,circle,rectangle):
    cv2.circle(image_file, circle['center'], int(circle['radius']), (255, 0, 255), 4)
    x,y,w,h = rectangle['bbox']
    x,y,x2,y2 = x,y,x+w,y+h
    cv2.rectangle(image_file, (x,y),(x2,y2) , (255, 0, 255), 4)
    plt.imshow(image_file)
    plt.show()
    

def get_x_y_coordinates(contour, image, padding=0):
    """
    Returns xmin, ymin, xmax, ymax with added padding
    """
    height, width = image.shape[:2]

    contour = np.array(contour)
    xmin = np.min(contour[:, 0])
    ymin = np.min(contour[:, 1])
    xmax = np.max(contour[:, 0])
    ymax = np.max(contour[:, 1])
    
    return xmin, ymin, xmax, ymax


def get_poly(contour): 
    
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    if len(approx) > 4 and len(approx) <= 14:
        return "cuboid", approx.squeeze()
    if len(approx) == 4:
        return "rectangle", approx.squeeze()
    return "undefined", approx.squeeze()


def get_Rectangles(edges,circle_info,image,remove_circle=True):
    
    rectangle_results = []
    
    # remove circle edges from circle_info
    if remove_circle:
        for circles in circle_info:
            if circles['shape'] == 'circle':
                cv2.circle(edges,circles['center'], int(circles['radius']+3),0,-1)
    
    # find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        name, poly = get_poly(cnt)
        
        if name == 'cuboid': 
            c = np.array(poly) 
            xmin = np.min(c[:, 0])
            ymin = np.min(c[:, 1])
            xmax = np.max(c[:, 0])
            ymax = np.max(c[:, 1])               
            rectangle_results.append({
                "shape"      : "rectangle",
                "contour"    : c,
                "pixel_area" : cv2.contourArea(c),
                "bbox"       : [xmin,ymin,xmax-xmin,ymax-ymin]
            })
            
        elif name == "rectangle":
            rectangle_results.append({
                "shape"      : "rectangle",
                "contour"    : poly,
                "pixel_area" : cv2.contourArea(poly),
                "bbox"       : [cv2.boundingRect(poly)]
            })
        
    return rectangle_results
            
            

def get_HoughCircles(edges, dp=1, minDist=500):
    circle_results = []
    
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist, param1=50, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            circle_meta = {
                "shape" : "circle",
                "Cx"    : int(x),
                "Cy"    : int(y),
                "radius": float(r),
                "center": [int(x), int(y)],
                "area"  : float(math.pi * r * r)
            }
            
            circle_results.append(circle_meta)
    return circle_results    

def get_edges(image, thresh1=1, thresh2=200 ):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    image = cv2.GaussianBlur(image, (3,3), 0)
    edges = cv2.Canny(image, thresh1, thresh2) # Canny Edge Detection
    return edges



def get_contours_info(image, thresh1=80, thresh2=200,dp=1, minDist=500):
    edges          = get_edges(image, thresh1, thresh2)
    circle_info    = get_HoughCircles(edges.copy(), dp, minDist)
    rectangle_info = get_Rectangles(edges.copy(),circle_info,image)
    
    return edges , circle_info , rectangle_info


def circle_attributes(circle, shape=(140,200), X=0.5, Y=0.5, Z=1):
    
    radius = circle['radius']
    h,w = shape
    
    radius = (radius / h if h>w else w) * Y if h>w else X
    
    
    surface_area = 4 * np.pi * radius * radius
    volume       = (np.pi * radius * radius * radius) * 4/3
    centroid     = [((circle['Cx'] / shape[1]) * X) , ((circle['Cy'] / shape[0]) * Y)]
    
    return {'name'         : 'circle',
            'surface_area' : surface_area,
            'volume'       : volume,
            'centroid'     : centroid
           }


def rectangle_attributes(rectangle, shape=(140,200), X=0.5, Y=0.5, Z=1):
    
    u,v,length,height = rectangle['bbox']
    
    # asssuming it might be 1.5 times per lenght
    width = (length if length > height else height) / 1.5 
    
    length = (length / shape[1]) * X
    height = (height / shape[0]) * Y 
    width  = (width  / shape[1]) * Z
    
    surface_area = (2 * length * width + 2 * length *  height + 2 * height * width)
    volume       = length * height * width
    centroid = [((u / shape[1]) * X) , ((v / shape[0]) * Y) ]
    
    return {'name'         : 'cuboid',
            'surface_area' : surface_area,
            'volume'       : volume,
            'centroid'     : centroid
           }

def get_attributes(circle, rectangle, shape=(140,200), X=0.5, Y=0.5, Z=1 ):
    
    meta_circle    = circle_attributes(circle, shape, X=0.5, Y=0.5, Z=1)
    meta_rectangle = rectangle_attributes(rectangle, shape, X=0.5, Y=0.5, Z=1)
    
    return [meta_circle,meta_rectangle]
    