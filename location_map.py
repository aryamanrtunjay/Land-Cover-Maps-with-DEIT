import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import pdb
import gmplot

root = "full_data_raw/rti_rwanda_crop_type/rti_rwanda_crop_type_source/"
root_label = "full_data_raw/rti_rwanda_crop_type/rti_rwanda_crop_type_labels/"

min_long = 1000000000
min_lat = 1000000000
max_long = -1000000000
max_lat = -1000000000

coordinates = []
bounding_box = []
for i in range(len(os.listdir(root)) - 10):
    filename = "rti_rwanda_crop_type_source_" + str(i)
    filename_root = "rti_rwanda_crop_type_labels_" + str(i)
    file_name = open(root + filename + "/" + filename + ".json")
    file_name_root = open(root_label + filename_root + "/vector_labels.json")
    label = json.load(file_name_root)["label"]
    
    data = json.load(file_name)
    bbox = data["bbox"]
    
    longs = []
    longs.append(bbox[0])
    longs.append(bbox[2])
    longitude = sum(longs)/2
    
    lats = []
    lats.append(bbox[1])
    lats.append(bbox[3])
    latitude = sum(lats)/2
    
    min_long = min(min_long, min(longs))
    min_lat = min(min_lat, min(lats))
    
    max_long = max(max_long, max(longs))
    max_lat = max(max_lat, max(lats))
    
    coordinates.append([label, longitude, latitude])
   
banana_lat = []
banana_long = []

legumes_lat = []
legumes_long = []

maize_lat = []
maize_long = []

structure_lat = []
structure_long = []

forest_lat = []
forest_long = []
for i in range(len(coordinates)):
    # Add code to add the coordinate to the correct list given its label
    if coordinates[i][0] == "banana":
        banana_lat.append(coordinates[i][2])
        banana_long.append(coordinates[i][1])
    elif coordinates[i][0] == "legumes":
        legumes_lat.append(coordinates[i][2])
        legumes_long.append(coordinates[i][1])
    elif coordinates[i][0] == "maize":
        maize_lat.append(coordinates[i][2])
        maize_long.append(coordinates[i][1])
    elif coordinates[i][0] == "structure":
        structure_lat.append(coordinates[i][2])
        structure_long.append(coordinates[i][1])
    elif coordinates[i][0] == "forest":
        forest_lat.append(coordinates[i][2])
        forest_long.append(coordinates[i][1])
    
bounding_box.append([min_long, min_lat, max_long, max_lat])
avg_lat = (max_lat + min_lat)/2
avg_long = (max_long + min_long)/2

gmap = gmplot.GoogleMapPlotter(avg_lat, avg_long, 10)
gmap.scatter(banana_lat, banana_long, '#FFFF00', size = 5, marker = True)
gmap.scatter(legumes_lat, legumes_long, '#0000FF', size = 5, marker = True)
gmap.scatter(maize_lat, maize_long, '#FF0000', size = 5, marker = True)
gmap.scatter(structure_lat, structure_long, '#FFA500', size = 5, marker = True)
gmap.scatter(forest_lat, forest_long, '#00FF00', size = 5, marker = True)
gmap.draw("map.html")