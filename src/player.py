import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


class Player():
    def __init__(self, color, side):
        self.color = color
        self.side = side
        self.points = 0
        # structures
        self.num_roads = 0
        self.num_towns = 0
        # cards
        self.num_clay = 0
        self.num_hay = 0
        self.num_stone = 0
        self.num_victory = 0
        self.num_wood = 0
        self.num_wool = 0
        
        
    #depends on card position
    def count_cards(self, cards):
        self.num_clay = 0
        self.num_hay = 0
        self.num_stone = 0
        self.num_victory = 0
        self.num_wood = 0
        self.num_wool = 0
        if side == "right":
            for card in cards:
                if card.track_window[x] > 540:
                    if card.type == "clay":
                        self.num_clay += 1 
                    elif card.type == "hay":
                        self.num_hay += 1
                    elif card.type == "stone":
                        self.num_stone += 1 
                    elif card.type == "victory":
                        self.num_victory += 1 
                    elif card.type == "wood":
                        self.num_wood += 1
                    elif card.type == "wool":
                        self.num_wool += 1 
        else:
            for card in cards:
                if card.track_window[x] < 540:
                    if card.type == "clay":
                        self.num_clay += 1 
                    elif card.type == "hay":
                        self.num_hay += 1
                    elif card.type == "stone":
                        self.num_stone += 1 
                    elif card.type == "victory":
                        self.num_victory += 1 
                    elif card.type == "wood":
                        self.num_wood += 1
                    elif card.type == "wool":
                        self.num_wool += 1
            

    #depends on player color
    def count_towns(self):
        pass

    def count_roads(self):
        pass