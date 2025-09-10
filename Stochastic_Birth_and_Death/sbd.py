import numpy as np
from PIL import Image
import cv2 as cv
from scipy.signal import convolve2d
from math import pi


# Euclidean distance between points (x1,y1), (x2,y2)
def l2(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


# Series approximation for inverse cosine
def acos(x):
    t1 = pi/2
    t2 = -x
    t3 = -(1/6)*x**3
    t4 = -(9/120)*x**5
    t5 = -(225/5040)*x**7
    t6 = -(11025/362880)*x**9
    t7 = -(893025/39916800)*x**11
    return t1+t2+t3+t4+t5+t6+t7


# Converts lon,lat coordinate pair to discretised matrix position
def coords_to_matrix(lon,lat, bounds, shape):
    
    h = shape[0]
    w = shape[1]

    lon_min = bounds[0][0]
    lon_max = bounds[1][0]
    lat_max = bounds[0][1]
    lat_min = bounds[1][1]

    row_temp = (lat_max - lat)/(lat_max-lat_min)
    col_temp = (lon - lon_min)/(lon_max-lon_min)

    row = int(np.round(row_temp*h))
    col = int(np.round(col_temp*w))

    return min(row,h-1), min(col,w-1)


# Converts matrix position to lon,lat coordinate pair
def matrix_to_coords(row, col, bounds, shape):
    
    h = shape[0]
    w = shape[1]
    
    
    lon_min = bounds[0][0]
    lon_max = bounds[1][0]
    lat_max = bounds[0][1]
    lat_min = bounds[1][1]

    lon = lon_min + col/w * (lon_max - lon_min)
    lat = lat_max - row/h * (lat_max - lat_min)
    
    return [lon, lat]


# Generates circular kernels of radii 1,...,Rmax
def generate_kernels(Rmax):
    
    kernels = []
    
    for r in range(1,Rmax+1):
        X = np.zeros([2*r+1, 2*r+1])
        for i in range(2*r+1):
            for j in range(2*r+1):
                if l2(i,j,r,r) <= r:
                    X[i,j] = 1
                    
        kernels.append(X)

    return kernels


# Pre-generates and stores data energy
def data(data_energy_filename = "data_energy.npz",bounds = [[-6.263564, 53.347969],[-6.246683, 53.339806]],
         osm = "osm_map.png", inter = "inter_data.csv", Rmax = 20):
    
    osm_map = (np.array((Image.open(osm)).convert("L")) != 255).astype(int)
    shape = osm_map.shape
    intersections = np.zeros([0,9])
    
    with open(inter,"r") as f:
        next(f)
        next(f)
        for line in f:
            x = line.split(",")
            d1,d2,lat,lon,del1,del2,c1,c2 = [float(y) for y in x]
            row, col = coords_to_matrix(lon, lat, bounds, shape)
            r = np.round(10*np.clip(1/d1+1/d2,0,1))
            intersections = np.vstack([intersections,np.array([row,col,d1,d2,del1,del2,c1,c2,int(r)])])
            
    X1 = np.zeros(shape)
    X2 = np.zeros(shape) 
    X3 = osm_map     
            
    for k in range(1,11):
        k_arr_1, k_arr_2 = np.zeros(shape), np.zeros(shape)
        k_mask = intersections[:,-1]==k
        k_intersections = intersections[k_mask]
        rows = k_intersections[:,0].astype(int)
        cols = k_intersections[:,1].astype(int)
        cnns = k_intersections[:,6]*k_intersections[:,7]
        dists = np.abs(k_intersections[:,2]-k_intersections[:,4])
        +np.abs(k_intersections[:,3]-k_intersections[:,5])
        k_arr_1[rows,cols] = cnns
        k_arr_2[rows,cols] = dists
        k1d = cv.getGaussianKernel(7, k)
        k2d = k1d@k1d.T
        k_arr_1 = convolve2d(k_arr_1, k2d, mode="same")
        k_arr_2 = convolve2d(k_arr_2, k2d, mode="same")
        X1+= k_arr_1
        X2+= k_arr_2
 
    kernels = generate_kernels(Rmax)
    pixel_count = [5.0, 13.0, 29.0, 49.0, 81.0, 113.0, 149.0, 197.0, 253.0, 317.0, 377.0, 441.0,
                   529.0, 613.0, 709.0, 797.0, 901.0, 1009.0, 1129.0, 1257.0]
    h,w = shape
    E0 = np.zeros([h,w,20])
    E1 = np.zeros([h,w,20])
    E2 = np.zeros([h,w,20])
    
    for r in range(1,Rmax+1):
        K = kernels[r-1]
        A = pixel_count[r-1]
        
        N0 = convolve2d(X1, K, mode="same")
        N1 = convolve2d(X2, K, mode="same")
        N2 = (1/A)*convolve2d(X3, K, mode="same")
        
        E0[:,:,r-1] = N0
        E1[:,:,r-1] = N1
        E2[:,:,r-1] = N2
    
    np.savez_compressed(data_energy_filename, array1=E0, array2=E1, array3=E2)
    
    return E0, E1, E2
       
  
# Main birth & death function
def birth_and_death(H, alpha, beta, delta, rb, rd, N0, Twait, Rmax, birth_decay=False):
    
    # pre-computed discretised area (pixels) of circles of radii 1,...,20
    pixel_count = [5.0, 13.0, 29.0, 49.0, 81.0, 113.0, 149.0, 197.0, 253.0, 317.0, 377.0, 441.0,
                   529.0, 613.0, 709.0, 797.0, 901.0, 1009.0, 1129.0, 1257.0] 
    
    # pre-compute pairwise energy penalties and store in lookup table
    pairwise_store = np.zeros([20,20,40])
    for r in range(1,21):
        for r_x in range(1,21):
            for dist in range(0,40):
                if dist<= np.abs(r-r_x):
                    E = min(pixel_count[r-1], pixel_count[r_x-1])/pixel_count[r-1]
                elif dist<r+r_x and dist>np.abs(r-r_x):
                    p1 = (r**2)*acos((dist**2 + r**2 - r_x**2)/(2*dist*r))
                    p2 = (r_x**2)*acos((dist**2 + r_x**2 - r**2)/(2*dist*r_x))
                    p3 = -0.5*np.sqrt((r+r_x-dist)*(r-r_x+dist)*(-r+r_x+dist)*(r+r_x+dist))
                    E = (p1+p2+p3)/pixel_count[r_x-1]
                elif dist>=r+r_x:
                    E = 0
                pairwise_store[int(r-1),int(r_x-1),int(dist)] = E

    energy_log = []
    energy_min = 0
    converged = False
    h,w,d = H.shape
    config = np.empty([0,5])   
    birth_log = np.empty([0,5])
    config_opt = config  
    counter1 = 0
    counter2 = 0
    
    while not converged:
        
        counter1 += 1
        
        ### Birth Step ###
        
        # choose radius
        r= int(np.round(np.random.exponential(10)))
        while r<1 or r>20:
            r= int(np.round(np.random.exponential(10)))
        
        # compute birthmap probabilities
        H_r = H[:,:,r-1]
        inds = np.argwhere(H_r<0)
        vals = -H_r[H_r<0]
        birth_probs = vals/(np.sum(vals))
        
        # choose number of points to spawn
        N = np.random.poisson(N0) if birth_decay == False else np.random.poisson(delta*N0)
        
        # randomly select points according to birth probabilities
        chosen_inds_idx = np.random.choice(len(inds), size=N, replace=False, p=birth_probs)
        chosen_inds = inds[chosen_inds_idx]
        
        temp_config = np.zeros([N,5])
        for n in range(N):
            i,j = chosen_inds[n,:]
            val = H_r[i,j]
            temp_config[n,:] = np.array([i,j,r,val,0])
            
        # add points to configuration
        config = np.vstack([config, temp_config])
        birth_log = np.vstack([birth_log, temp_config])
        
        ### Death Step ###
        
        # sort current points according to unary energy
        score = config[:, 3]
        sorted_indices = np.argsort(-score)  
        config = config[sorted_indices]
        
        # compute pairwise penalties
        i_coords = config[:, 0]
        j_coords = config[:, 1]
        r_vals   = config[:, 2]
        
        i1 = i_coords[:, np.newaxis]  
        j1 = j_coords[:, np.newaxis]
        r1 = r_vals[:, np.newaxis]

        i2 = i_coords[np.newaxis, :]  
        j2 = j_coords[np.newaxis, :]
        r2 = r_vals[np.newaxis, :]
        
        dist = np.sqrt((i1 - i2)**2 + (j1 - j2)**2)  
        dist_idx = dist.astype(int)
        
        valid_mask = dist < 40
        safe_dist_idx = dist_idx.copy()
        safe_dist_idx[~valid_mask] = -1
        
        r1_idx = np.broadcast_to(r1, dist_idx.shape).astype(int) 
        r2_idx = np.broadcast_to(r2, dist_idx.shape).astype(int)
        
        L = pairwise_store[r1_idx - 1, r2_idx - 1, safe_dist_idx]
        L[~valid_mask] = 0
        np.fill_diagonal(L, 0)
        
        # iterate through sorted configuration and kill points
        row=0
        Lrow=0
        N = config.shape[0]
        while Lrow<N:
            pairwise = np.sum(L[Lrow,:])
            point = config[row,:]
            E = point[3] + alpha*pairwise
            a = np.exp(beta*E)
            d = (delta*a)/(1+delta*a)
            
            if np.random.uniform(0,1)<d:
                config = np.delete(config, row, axis=0)
                L[:,Lrow]=0
                L[Lrow,:]=0
            else:
                row+=1
                
            Lrow+=1
        
        # check if new minimal energy has been found
        total_energy = np.sum(config[:,3]) + np.sum(L)
        energy_log.append(total_energy)
        
        if total_energy<=energy_min:
            energy_min = total_energy
            config_opt = config
            counter2=0
        else:
            counter2+=1
        
        # report - currently every 200 iterations
        if counter1%200 == 0:
            print("Iteration "+str(counter1)+" complete. Minimal energy: "+str(energy_min))
        
        # update parameters (geometric annealing)
        beta*=rb
        delta*=rd
        
        # check for convergence or max number of iterations reached
        if counter2>Twait or counter1>10000:
            converged = True
            
    return config_opt, birth_log
        

# Converts output of birth & death to lon,lat coordinates and saves to .csv file
def convert_and_save(save_filename, config, bounds, shape):
    
    with open(save_filename, "w") as f:
        f.write("{0:s},{1:s},{2:s},{3:s}\n".format("lon", "lat", "rad", "data"))
        f.close()
        
    for row in config:
        i, j, r, e, p = row
        lon, lat = matrix_to_coords(i, j, bounds, shape)
        with open(save_filename, "a") as f:
            f.write("{0:f},{1:f},{2:f},{3:f}\n".format(lon,lat,r,e))
            f.close()   
    return
        
        
# Controls simulation including parameters, birth & death process, and file writing  
def simulation(bounds, osm_filename, inter_filename, data_energy_filename, 
               W, alpha, beta, delta, rb, rd, N0, Twait, Rmax, new_data, birth_decay):
    
   
    if new_data == True:
        print("Generating data")
        E0, E1, E2 = data(data_energy_filename, bounds, osm_filename, inter_filename, Rmax)
        print("Data generated successfully and saved to .npz file")
    else:
        print("Importing data from .npz file")
        arrays = np.load(data_energy_filename)
        E0 = arrays["array1"]
        E1 = arrays["array2"]
        E2 = arrays["array3"]
        
    H = W[0]*E0 + W[1]*E1 + W[2]*E2
    
    shape = [H.shape[i] for i in [0,1]]
    
    print("Birth and death process initiated")
    config, birth_log = birth_and_death(H, alpha, beta, delta, rb, rd, N0, Twait, Rmax, birth_decay)
    print("Birth and death process converged")
    print("Saving optimal configuration")
    convert_and_save("solution.csv", config, bounds, shape)
    convert_and_save("birth_log.csv",birth_log, bounds, shape)
    print("Configuration saved")

    return config   


### example inputs for use with github files ###
bounds = [[-6.263564, 53.347969],[-6.246683, 53.339806]] # top-left and bottom-right lon,lat coordinates of map area
osm_filename = "osm_map.png"                             # osm information stored as image
inter_filename = "inter_data.csv"                        # file containing information on pairwise intersections
data_energy_filename = "data_energy.npz"                 # data energy - to save or load


### parameters ###
W = [-1,0.1,0.4]    # weights between unary energy terms
alpha = 7           # weight between unary and pairwise energy terms 
beta = 1            # initial inverse temperature parameter
delta = 1           # initial discretisation parameter
rb = 1.001          # rate of increase of inverse temperature (geometric annealing)
rd = 0.999          # rate of decrease of discretisation (geometric annealing)
N0 = 100            # number of objects to birth per iteration (relates to object density)
Twait = 500         # end simulation Twait iterations with no new minimum
Rmax = 20           # largest object radius to be considered (pixels), data must be regenerated if increased
new_data = True     # set to false once data has been generated and saved
birth_decay = False # disables annealing on birth step for aggressive exploration


### run simulation ###           
simulation(bounds, osm_filename, inter_filename, data_energy_filename,
           W, alpha, beta, delta, rb, rd, N0, Twait, Rmax, new_data, birth_decay)        
        
        
        
        
        
        
        
        
        
        