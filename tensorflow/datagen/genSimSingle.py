#******************************************************************************
#
# Varying density data gen, 2d/3d single sim
#
#******************************************************************************

from manta import *
import os, shutil, math, sys, time
from datetime import datetime
import numpy as np
sys.path.append("../tools")
import paramhelpers as ph

# Main params  ----------------------------------------------------------------------#
steps    = 1000
simNo    = 1000  # start ID
showGui  = 0
basePath = '../train_data/'
npSeedstr   = "-1"
dim         = 2 

# Solver params  
res         = 64

# cmd line
basePath        =     ph.getParam( "basepath",        basePath        )
npSeedstr       =     ph.getParam( "seed"    ,        npSeedstr       )
npSeed          =     int(npSeedstr)
dim   			= int(ph.getParam( "dim"     ,        dim))
savenpz 		= int(ph.getParam( "savenpz",         False))>0
saveuni 		= int(ph.getParam( "saveuni",         False))>0
saveppm 		= int(ph.getParam( "saveppm" ,        False))>0
showGui 		= int(ph.getParam( "gui"     ,        showGui))
res     		= int(ph.getParam( "res"     ,        res))
steps     		= int(ph.getParam( "steps"   ,        steps))
setDt 	    	= float(ph.getParam( "dt"   ,         0.1))
frameLen        = float(ph.getParam( "frameLen"  ,    3.0))
timeOffset   	= int(ph.getParam( "warmup"  ,        5))    # skip certain no of frames at beginning
ph.checkUnusedParams()

setDebugLevel(1)
if not basePath.endswith("/"): basePath = basePath+"/"

if savenpz or saveuni or saveppm: 
	folderNo = simNo
	simPath,simNo = ph.getNextSimPath(simNo, basePath)

	# add some more info for json file
	ph.paramDict["simNo"] = simNo
	ph.paramDict["type"] = "smoke"
	ph.paramDict["name"] = "gen6combined"
	ph.paramDict["version"] = printBuildInfo()
	ph.paramDict["creation_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
	ph.writeParams(simPath + "description.json") # export sim parameters 

	sys.stdout = ph.Logger(simPath)
	print("Called on machine '"+ os.uname()[1] +"' with: " + str(" ".join(sys.argv) ) )
	print("Saving to "+simPath+", "+str(simNo))
	# optional , backupFile(__file__, simPath)  

if(npSeed<0): 
	npSeed = np.random.randint(0, 2**32-1 )
print("Random seed %d" % npSeed)
np.random.seed(npSeed)

# Init solvers -------------------------------------------------------------------#
gridSize = vec3(res,res,res) 
if (dim==2): gridSize.z = 1  # 2D

buoyFac = 0.1 + np.random.rand()
if buoyFac < 0.4:
	buoyFac = 0. # make sure we have some sims without buoyancy
buoy = vec3(0,-3e-3*0.25,0) * buoyFac
print("Buoyancy: " + format(buoy) +", factor " + str(buoyFac))

# solvers
solver = Solver(name='smaller', gridSize = gridSize, dim=dim)
solver.timestep = setDt
solver.frameLength = frameLen
timings = Timings()

# Simulation Grids  -------------------------------------------------------------------#
flags    = solver.create(FlagGrid)
vel      = solver.create(MACGrid)
density  = solver.create(RealGrid)
pressure = solver.create(RealGrid)
vorticityTmp = solver.create(Vec3Grid) # vorticity, optional
norm     = solver.create(RealGrid)

if savenpz:
	sm_arR = np.zeros([int(gridSize.z), int(gridSize.y), int(gridSize.x), 1])
	sm_arV = np.zeros([int(gridSize.z), int(gridSize.y), int(gridSize.x), 3])
	sm_arP = np.zeros([int(gridSize.z), int(gridSize.y), int(gridSize.x), 1])

# open boundaries
bWidth=1
flags.initDomain(boundaryWidth=bWidth)
flags.fillGrid()

setOpenBound(flags,    bWidth,'yY',FlagOutflow|FlagEmpty) 

# inflow sources ----------------------------------------------------------------------#

# init random density
sources   = []
noise     = []  
inflowSrc = [] # list of IDs to use as continuous density inflows

noiseN = 12
nseeds = np.random.randint(10000,size=noiseN)

cpos = vec3(0.5,0.5,0.5)

randoms = np.random.rand(noiseN, 10)
for nI in range(noiseN):
	
	noise.append( solver.create(NoiseField, fixedSeed= int(nseeds[nI]), loadFromFile=True) )
	noise[nI].timeAnim = 0.3
	noise[nI].posOffset = vec3(1.5) 
	noise[nI].posScale = vec3( res * 0.1 * (randoms[nI][7] + 1) ) 
	noise[nI].clamp = True
	noise[nI].clampNeg = 0
	noise[nI].clampPos = 1.0
	noise[nI].valScale = 1.0
	noise[nI].valOffset = 0.5 * randoms[nI][9]
	
	# random offsets
	coff = vec3(0.4) * (vec3( randoms[nI][0], randoms[nI][1], randoms[nI][2] ) - vec3(0.5))
	radius_rand = 0.035 + 0.035 * randoms[nI][3]
	upz = vec3(0.95)+ vec3(0.1) * vec3( randoms[nI][4], randoms[nI][5], randoms[nI][6] )

	if 1 and randoms[nI][8] > 0.5: # turn into inflow?
		if coff.y > -0.2:
			coff.y += -0.4
		coff.y *= 0.5
		inflowSrc.append(nI)

	if(dim == 2): 
		coff.z = 0.0
		upz.z = 1.0
	sources.append( solver.create(Sphere, center=gridSize*(cpos+coff), radius=gridSize.x*radius_rand, scale=upz))
		
	print (nI, "centre", gridSize*(cpos+coff), "radius", gridSize.x*radius_rand, "other", upz )	
	densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )

# init random velocities
inivel_sources , inivel_vels    = [] , []
if 1: # from fluidnet
	c = 3 + np.random.randint(3) # "sub" mode

	# 3..5 - ini vel sources
	if c==3: numFac = 1; sizeFac = 0.9
	if c==4: numFac = 3; sizeFac = 0.7
	if c==5: numFac = 5; sizeFac = 0.6

	numNs = int( numFac * float(dim) )  
	for ns in range(numNs):
		p = [0.5,0.5,0.5]
		Vrand = np.random.rand(10) 
		for i in range(3):
			p[i] += (Vrand[0+i]-0.5) * 0.6
		p = Vec3(p[0],p[1],p[2])
		size = ( 0.05 + 0.1*Vrand[3] ) * sizeFac

		v = [0.,0.,0.]
		for i in range(3):
			v[i] -= (Vrand[0+i]-0.5) * 0.6 * 2. # invert pos offset , towards middle
			v[i] += (Vrand[4+i]-0.5) * 0.3      # randomize a bit, parametrized for 64 base
		v = Vec3(v[0],v[1],v[2])
		v = v*0.9*0.5 # tweaking
		v = v*(1. + 0.5*Vrand[7] ) # increase by up to 50% 

		print( "IniVel Pos " + format(p) + ", size " + format(size) + ", vel " + format(v) )
		sourceVsm = solver.create(Sphere, center=gridSize*p, radius=gridSize.x*size, scale=vec3(1))
		inivel_sources.append(sourceVsm)
		inivel_vels.append(v) 

# Setup UI ---------------------------------------------------------------------#
if (showGui and GUI):
	gui=Gui()
	gui.show()
	gui.pause()

t = 0
doPrinttime = False

# main loop --------------------------------------------------------------------#
lastFrame = -1
while solver.frame < steps+timeOffset:
	curt = t * solver.timestep
	sys.stdout.write( "Current sim time t: " + str(curt) +" \n" )
	#density.setConst(0.); # debug reset

	if doPrinttime:
		starttime = time.time()
		print("starttime: %2f" % starttime)	

	if 1 and len(inflowSrc)>0:
		# note - the density inflows currently move with the offsets!
		for nI in inflowSrc:
			densityInflow( flags=flags, density=density, noise=noise[nI], shape=sources[nI], scale=1.0, sigma=1.0 )
	
	advectSemiLagrange(flags=flags, vel=vel, grid=vel, order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)
	setWallBcs(flags=flags, vel=vel)
	addBuoyancy(density=density, vel=vel, gravity=buoy , flags=flags)

	for i in range(len(inivel_sources)):
		inivel_sources[i].applyToGrid( grid=vel , value=inivel_vels[i] )	

	if 1 and ( curt < timeOffset ): 
		vorticityConfinement( vel=vel, flags=flags, strength=0.05 )

	solvePressure(flags=flags, vel=vel, pressure=pressure,  cgMaxIterFac=2.0, cgAccuracy=0.001, preconditioner=PcMGStatic )
	setWallBcs(flags=flags, vel=vel)
	advectSemiLagrange(flags=flags, vel=vel, grid=density, order=2, clampMode=2, openBounds=True, boundaryWidth=bWidth)

	if doPrinttime:
		endtime = time.time()
		print("endtime: %2f" % endtime)
		print("runtime: %2f" % (endtime-starttime))

	# --------------------------------------------------------------------#

	# save low and high res
	# save all frames
	if solver.frame >= timeOffset and lastFrame!=solver.frame:
		tf = solver.frame - timeOffset
		if savenpz:
			print("Writing NPZs for frame %d"%tf)
			copyGridToArrayReal( target=sm_arR, source=density )
			np.savez_compressed( simPath + 'density_low_%04d.npz' % (tf), sm_arR )
			copyGridToArrayVec3( target=sm_arV, source=vel )
			np.savez_compressed( simPath + 'velocity_low_%04d.npz' % (tf), sm_arV )

			#save pressure field
			copyGridToArrayReal( target=sm_arP, source=pressure)
			np.savez_compressed( simPath + 'pressure_low_%04d.npz' % (tf), sm_arP )


		if saveuni:
			print("Writing UNIs for frame %d"%tf)
			density.save(simPath + 'density_low_%04d.uni' % (tf))
			vel.save(simPath + 'velocity_low_%04d.uni' % (tf))
			#computeVorticity(vel = vel,vorticity = vorticityTmp,norm = norm) #vorticity
			#vorticityTmp.save(simPath + 'vorticity_low_%04d.uni' % (tf))

			pressure.save(simPath + 'pressure_low_%04d.uni' % (tf))
			
		if(saveppm):
			print("Writing ppms for frame %d"%tf)
			projectPpmFull( density, simPath + 'density_low_%04d.ppm' % (tf), 0, 2.0 )

	lastFrame = solver.frame
	solver.step()
	#gui.screenshot( 'out_%04d.jpg' % t ) 
	#timings.display() 
	t = t+1

