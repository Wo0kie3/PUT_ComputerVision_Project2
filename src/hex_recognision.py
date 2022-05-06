import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class Hex():
    def __init__(self, circle, boundary, type):
        self.circle = circle
        self.boundary = boundary
        self.type = type 

    def __str__(self):
        return "{} {} {}".format(self.circle, self.boundary, self.type)



def image_distance_knn(query_desc, target_desc, distance_measure, knn):
    if query_desc is None or target_desc is None:
        return 9999
    bf_matcher = cv2.BFMatcher(distance_measure)
    matches = bf_matcher.knnMatch(query_desc, target_desc, k=knn)
    matches = [np.mean([x.distance for x in kp_matches]) for kp_matches in matches]
    dist_mean = np.mean(matches, axis = 0)
    return dist_mean


def hex_distance(query_img, hex_dataset, detector, descriptor, distance_measure, knn = 1):
    query_kp = detector.detect(query_img, None)
    _, query_desc = descriptor.compute(query_img, query_kp)

    outputs = []

    for hex_name, target_img in hex_dataset.items():
        target_img = cv2.resize(target_img, query_img.shape)
        target_kp = detector.detect(target_img, None)
        _, target_desc = descriptor.compute(target_img, target_kp)
        dist_mean = image_distance_knn(query_desc, target_desc, distance_measure, knn)
        outputs.append({'hex': hex_name, 'dist_mean': dist_mean})


    outputs.sort(key=lambda x: x['dist_mean'])
    return outputs

def hex_distance_color(query_img, hex_dataset):
    outputs = []

    query_means = np.median(query_img, axis=(0, 1))
    for hex_name, target_img in hex_dataset.items():
        target_means = np.median(target_img, axis=(0, 1))
        dist_mean = np.linalg.norm(target_means - query_means, ord = 2)
        outputs.append({'hex': hex_name, 'dist_mean': dist_mean})

    outputs.sort(key=lambda x: x['dist_mean'])
    return outputs


def getHexCircles(frame, expected_radius : int = 60):
    min_radius = int(expected_radius * 0.85)
    max_radius = int(expected_radius * 1.15)
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, 100, param1=75, param2=20, minRadius=min_radius, maxRadius=max_radius)
    circles = np.round(circles[0, :]).astype("int")
    return circles


def compareHexes(frame, circles, expected_radius : int = 60, template_path = './img/hexes'):
    hex_names = ['desert', 'field', 'forest', 'mountain', 'pasture', 'quarry']
    hex_templates = dict()

    for hex_name in hex_names:
        hex_templates[hex_name] = cv2.imread('{}/{}.jpg'.format(template_path, hex_name), cv2.IMREAD_COLOR)
        hex_templates[hex_name] = cv2.cvtColor(hex_templates[hex_name], cv2.COLOR_BGR2LAB)[..., 0]
    sift =  cv2.SIFT_create()

    query_images = retrieveHexSegments(frame, circles, expected_radius)
    results = []
    for idx, query_img in enumerate(query_images):
        img = query_img['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
        results.append({'img': img, 'distance': hex_distance(img, hex_templates, sift, sift, cv2.NORM_L2, 1), 'circle': query_img['circle']})
    return results


def retrieveHexSegments(frame, circles, expected_radius : int = 60):
    segments = []
    for (x, y, r) in circles:
        min_x = max(0, x - expected_radius)
        min_y = max(0, y - expected_radius)
        max_x = min(frame.shape[1], x + expected_radius)
        max_y = min(frame.shape[0], y + expected_radius)

        segments.append({'img': frame[min_y : max_y, min_x : max_x], 'circle': (x, y, r)})
    return segments


def retrieveHexInfo(frame, expected_radius : int = 60, expected_distance : int = 220, template_path = './img/hexes'): 

    frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[..., 0]

    circles = getHexCircles(frame_lab, expected_radius)
    
    # find the largest cluster of "hexes"
    aglo = AgglomerativeClustering(n_clusters=None, linkage='single', distance_threshold = expected_distance)
    classes = aglo.fit_predict([(item[0], item[1]) for item in circles])

    class_count = {}
    for cls in classes:
        if cls in class_count.keys():
            class_count[cls] += 1
        else:
            class_count[cls] = 1

    popular_class = max(class_count, key=class_count.get)
    new_circles = []

    for idx, cls in enumerate(classes):
        if cls == popular_class:
            new_circles.append(circles[idx])
        
    hexes_distance = compareHexes(frame, new_circles, expected_radius, template_path) 

    hexes = []
    for hex in hexes_distance:
        (x, y, r) = hex['circle']
        hex_type = hex['distance'][0]['hex']
        min_x = max(0, x - r)
        min_y = max(0, y - r)
        w = min(2 * r, frame_lab.shape[1] - min_x)
        h = min(2 * r, frame_lab.shape[0] - min_y)

        new_hex = Hex(hex['circle'], (min_x, min_y, w, h), hex_type)
        hexes.append(new_hex)

    return hexes