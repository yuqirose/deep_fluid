# -*- coding: utf-8 -*-
import os, math, sys
import scipy.misc, numpy as np
import numpy as np
from os.path import basename
sys.path.append("../tools")
import paramhelpers as ph
from manta import *

file_type = r"npz"
folder = '/home/roseyu/Documents/manta/saves/seq2seq/'# npz file path
folder  = ph.getParam("folder", folder)
print("Converting ",file_type, " to ppm!")


sz = vec3( 64, 64, 64) # simulation resolution
sz.z  = 1 #2D
Sl = Solver(name='sl', gridSize = sz, dim=2) 
Gd = Sl.create(RealGrid)
for root, dirs, files in os.walk(folder): 
    for f in sorted(files):
        if f.endswith(".npz")<=0: continue
        filepath = os.path.join(root, f)
        nparray = np.copy(np.load(filepath)["arr_0"]) # we usually use "arr_0", np copy is usually good to avoid non-standard view
        # check image/video
        n_imgs = nparray.shape[0]
        for i in range(n_imgs):
            copyArrayToGridReal(nparray[i,], Gd)
            new_fname = os.path.splitext(basename(f))[0]+"_"+str(i)+".ppm"
            newfile = os.path.join(root, new_fname) # use the same name
            projectPpmFull( Gd, newfile, 0, 1.7 )
