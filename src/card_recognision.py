import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class Card():
    def __init__(self, rectangle, track_window, type):
        self.rectangle = rectangle
        self.track_window = track_window
        self.type = type 

    def __str__(self):
        return "{} {} {}".format(self.rectangle, self.track_window, self.type)



def image_distance_knn(query_desc, target_desc, distance_measure, knn):
    if query_desc is None or target_desc is None:
        return 9999
    bf_matcher = cv2.BFMatcher(distance_measure)
    matches = bf_matcher.knnMatch(query_desc, target_desc, k=knn)
    matches = [np.mean([x.distance for x in kp_matches]) for kp_matches in matches]
    dist_mean = np.mean(matches, axis = 0)
    return dist_mean


def card_distance(query_img, hex_dataset, detector, descriptor, distance_measure, knn = 1):
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


def card_distance_color(query_img, hex_dataset):
    outputs = []

    query_means = np.median(query_img, axis=(0, 1))
    for hex_name, target_img in hex_dataset.items():
        target_means = np.median(target_img, axis=(0, 1))
        dist_mean = np.linalg.norm(target_means - query_means, ord = 2)
        outputs.append({'hex': hex_name, 'dist_mean': dist_mean})

    outputs.sort(key=lambda x: x['dist_mean'])
    return outputs


def getCardRectangles(frame):
    thr = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1]
    (t, thr) = cv2.threshold(thr, 65, 255, cv2.THRESH_BINARY)

    countr = cv2.findContours(thr, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    rectangles = []
    track_windows = []

    for cont in countr:
        area = cv2.contourArea(cont)
        if area > 2000 and area < 10000:
            rect = cv2.minAreaRect(cont)
            (mid_x, mid_y), (w, h), r = rect
            proportion = max(w, h) / min(w, h)
            if proportion < 1.4 or proportion > 1.6:
                continue
            rectangles.append(rect)
            track_windows.append(cv2.boundingRect(cont))

    return rectangles, track_windows

def retrieveCardSegments(frame, rectangles, track_windows):
    segments = []
    for rect, track_window in zip(rectangles, track_windows):
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_img = cv2.warpPerspective(frame, M, (width, height))

        segments.append({'img': warped_img, 'rectangle': rect, 'track_window': track_window})
    return segments


def compareCards(frame, rectangles, track_windows, template_path = './img/cards'):
    hex_names = ['clay', 'hay', 'stone', 'victory', 'wood', 'wool']
    hex_templates = dict()

    for hex_name in hex_names:
        hex_templates[hex_name] = cv2.imread('{}/{}.jpg'.format(template_path, hex_name), cv2.IMREAD_COLOR)
        hex_templates[hex_name] = cv2.cvtColor(hex_templates[hex_name], cv2.COLOR_BGR2LAB)[..., 0]
    sift =  cv2.SIFT_create()

    query_images = retrieveCardSegments(frame, rectangles, track_windows)
    results = []
    for idx, query_img in enumerate(query_images):
        img = query_img['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
        results.append({'img': img, 'distance': card_distance(img, hex_templates, sift, sift, cv2.NORM_L2, 1), 'rectangle': query_img['rectangle'], 'track_window': query_img['track_window']})
    return results


def retrieveCardInfo(frame, template_path = './img/cards'): 

    rectangles, track_windows = getCardRectangles(frame)
        
    card_distance = compareCards(frame, rectangles, track_windows, template_path) 

    cards = []
    for card in card_distance:
        card_type = card['distance'][0]['hex']
        new_card = Card(card['rectangle'], card['track_window'], card_type)
        cards.append(new_card)

    return cards