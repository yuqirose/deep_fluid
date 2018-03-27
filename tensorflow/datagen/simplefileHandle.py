import os, math, sys
import scipy.misc, numpy as np

test = 2
print("Test case %d!" % test)

def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img,img,img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img
   
def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)
    
def save_img_3d(out_path, img): # img: e.g.,(56,64,48), a 3D float/double array
    dimY = max(img.shape[0], img.shape[1]) # max of y and z
    imgXY = np.sum(img, axis=0) # shape[0] : y
    imgZY = np.transpose(np.sum(img, axis=2))# shape[0] : y
    imgXZ = np.sum(img, axis=1)# shape[0] : z
    if(img.shape[0] < dimY):
        imgXZ = np.pad( imgXZ, ((0,dimY - img.shape[0]),(0,0)) )
    elif(img.shape[1] < dimY):
        imgXY = np.pad( imgXY, ((0,dimY - img.shape[1]),(0,0)) )
        imgZY = np.pad( imgZY, ((0,dimY - img.shape[1]),(0,0)) )
    
    data = np.concatenate([imgXY, imgZY, imgXZ], axis=1 )
    save_img( out_path, data)
    
if( test == 1 ): # uni simple rendering
    from manta import *
    folder = r"/home/roseyu/Documents/manta/saves/" # uni file path
    sz = vec3( 64, 64, 64) # simulation resolution
    sz.z  = 1 # 2D
    Sl = Solver(name='sl', gridSize = sz, dim=2) 
    Gd = Sl.create(RealGrid)
    for root, dirs, files in os.walk(folder): 
        for f in sorted(files):
            if f.endswith(".uni")<=0: continue
            filepath = os.path.join(root, f)
            Gd.load(filepath)
            newfile = os.path.join(root, f.replace(".uni", ".ppm")) # use the same name
            projectPpmFull( Gd, newfile, 0, 1.7 ) 
            # projectPpmFull ( Grid< Real >&val,   -> grid
            #    string name,                      -> image path + name
            #    int shadeMode = 0,                -> shading modes: 0 smoke, 1 surfaces
            #    Real scale = 1.                   -> scaling the grid value, 2D grids usually use 1.0, 3D grids use larger ones. 
            #    )	
            
elif( test == 2 ): # NPZ simple rendering
    from manta import *
    folder = r"/home/roseyu/Documents/manta/saves/seq2seq/"# npz file path
    sz = vec3( 64, 64, 64) # simulation resolution
    sz.z  = 1 #2D
    Sl = Solver(name='sl', gridSize = sz, dim=2) 
    Gd = Sl.create(RealGrid)
    for root, dirs, files in os.walk(folder): 
        for f in sorted(files):
            if f.endswith(".npz")<=0: continue
            filepath = os.path.join(root, f)
            nparray = np.copy(np.load(filepath)["arr_0"]) # we usually use "arr_0", np copy is usually good to avoid non-standard view
            copyArrayToGridReal(nparray, Gd)
            newfile = os.path.join(root, f.replace(".npz", ".ppm")) # use the same name
            projectPpmFull( Gd, newfile, 0, 1.7 )

elif( test == 3 ): # simply sum NPZ along one dim, no manta is needed
    folder = r"D:\code\Gan_related\mantaGan\tensorflow\data_sim\3D_fullframes_density\sim_1094"
    for root, dirs, files in os.walk(folder): 
        for f in sorted(files):
            if f.endswith(".npz")<=0: continue
            filepath = os.path.join(root, f)
            nparray = np.load(filepath)["arr_0"] # we usually use "arr_0"
            newfile = filepath.replace(".npz", ".png") # use the same name
            # we usually save a 4D array, e.g., (256,256,256,1)
            if(len(nparray.shape) == 4):
                if(nparray.shape[0] != 1) : # our 3D data
                    save_img_3d( newfile, nparray[:,:,:,0] * 20. )
                else: # our 2D data
                    save_img( newfile, nparray[0][:,:,:,0] * 255. )
            elif(len(nparray.shape) == 3):
                save_img( newfile, nparray[:,:,0] * 255. )
