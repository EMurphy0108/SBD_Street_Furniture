from math import radians, cos, sin, asin, sqrt, atan2, degrees
import numpy as np
from sklearn.neighbors import NearestNeighbors


# haversine distance between two points
def haversine(lon1, lat1, lon2, lat2, CoordThreshold = 0.001):
    if abs(lon1 - lon2) > CoordThreshold:
        return 100
    if abs(lat1 - lat2) > CoordThreshold:
        return 100
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    return 6367000. * sqrt( ((lon2-lon1) * cos(0.5*(lat2+lat1)))**2 + (lat2-lat1)**2);


# calculates bearing from pointA to pointB
def calculate_bearing(pointAlat,pointAlon, pointBlat,pointBlon):
    lat1 = radians(pointAlat)
    lat2 = radians(pointBlat)
    diffLong = radians(pointBlon - pointAlon)
    x = sin(diffLong) * cos(lat2)
    y = cos(lat1) * sin(lat2) - (sin(lat1)* cos(lat2) * cos(diffLong))
    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    return compass_bearing


# uses 'dst' and 'bear' to project new point from 'lon,lat'
def ShiftLonLat(lon,lat,bear,dst):
    R = 6378.1 #Radius of the Earth
    brng = radians(bear)
    d = .001*dst

    lat1 = radians(lat) #Current lat point converted to radians
    lon1 = radians(lon) #Current long point converted to radians

    lat2 = asin( sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(brng))
    lon2 = lon1 + atan2(sin(brng)*sin(d/R)*cos(lat1),cos(d/R)-sin(lat1)*sin(lat2))
    return degrees(lon2), degrees(lat2)


# proposes noisy/contaminated detections given asset_file and image_file
# iterates through street-view images and simulates detections of nearby assets
# can simulate contaminate (false positive) detections
def propose_detections(asset_file, image_file, write_file = "detections.csv", contaminations = True,
                       contamination_level = 0.05, prob1 = 0.7, prob2 =0.9, sd_dist = 2, sd_bear = 3):
    
    detections = []
    counter = 0
    
    with open(asset_file, "r") as f:
        asset_content = f.readlines()
        asset_content = [[float(row.split(",")[0]), float(row.split(",")[1][:-1])] for row in asset_content]
        f.close()
        
    with open(image_file, "r") as f:
        image_content = f.readlines()
        image_content = [[float(row.split(",")[0]), float(row.split(",")[1][:-1])] for row in image_content]
        f.close()
       
    for x in asset_content:
        pt = np.array(x).reshape(1,-1) 
        n = NearestNeighbors(n_neighbors = 15, algorithm='auto').fit(asset_content)
        distances, indices = n.kneighbors(pt)
        
        for i in indices[0]:
            pt2 = asset_content[i]
            pt1 = x
            dist = haversine(pt1[0],pt1[1],pt2[0],pt2[1])
        
            a = np.random.uniform(0,1)      
    
            if (dist<2 or (dist>10 and dist<20)) and a<prob1:           
                detect = True
            elif dist>=2 and dist<=10 and a<prob2:                      
                detect = True
            else:
                detect = False
                
            if detect == True:
                counter+=1
                D = np.random.normal(dist, sd_dist)
                while D<=0:
                    D = np.random.normal(dist, sd_dist)
                bear = calculate_bearing(pt1[1],pt1[0], pt2[1],pt2[0])  
                B = np.random.normal(bear, sd_bear) 
                
                pt3 = ShiftLonLat(pt1[0],pt1[1],B,D)
                CNN_prob = max(1 - np.random.exponential(0.125),0.5)
                
                detections.append([pt3[0], pt3[1], D, CNN_prob,0])
                
    if contaminations == True:
        n = int(np.round(contamination_level*counter))
        for i in range(n):
            ind = int(np.random.uniform(0,len(image_content)))
            pt = image_content[ind]
            D = np.random.uniform(1,15)
            B = np.random.uniform(0,360)
            CNN_prob = max(1 - np.random.exponential(0.125),0.5)
            pt3 = ShiftLonLat(pt[0],pt[1],B,D)
            detections.append([pt3[0], pt3[1], D, CNN_prob, 1])
    
    with open(write_file, "w") as f:
        f.write("{0:s},{1:s},{2:s},{3:s},{4:s}\n".format("lon", "lat", "Depth", "CNN", "Contaminate?"))
        for x in detections:
            f.write("{0:f},{1:f},{2:f},{3:f},{4:f}\n".format(x[0], x[1], x[2], x[3], x[4]))
            
    return


### parameters ###
contamination_level = 0.05 # proportion of false positive detections relative to real detections
p1 = 0.7                   # probability of detection from distances in [0,2) u (10,20)
p2 = 0.9                   # probability of detection from distances in [2,10]
sd_dist = 2                # standard deviation on distance noise (metres)
sd_bear = 3                # standard deviation on bearing noise (degrees)

### run main function ###
propose_detections("assets.csv", "street_images.csv", contamination_level, p1, p2, sd_dist, sd_bear)
    
    