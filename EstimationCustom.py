'''
Main script for Orbit Determination.

Authors:    James Johnson

Created:    Feb-2020
'''

#Initialize OREkit and Java Virtual Machine
import orekit
vm = orekit.initVM()
#Set up OREkit Directory
from orekit.pyhelpers import setup_orekit_curdir, JArray_double2D
print('[00:00.00] | Loading Orekit Virtual Machine')
setup_orekit_curdir()


#Import Orekit Libraries
from org.orekit.data import DataProvidersManager, DirectoryCrawler
from org.orekit.frames import FramesFactory, ITRFVersion, TopocentricFrame, LocalOrbitalFrame, LOFType, Transform
from org.orekit.utils import IERSConventions, Constants, CartesianDerivativesFilter
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.propagation.analytical.tle import TLE
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.attitudes import NadirPointing
from org.orekit.propagation.analytical.tle import SGP4
from org.orekit.orbits import CartesianOrbit, OrbitType, PositionAngle
from org.orekit.propagation.conversion import DormandPrince853IntegratorBuilder, NumericalPropagatorBuilder
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient, SolarRadiationPressure
from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
from org.orekit.models.earth.atmosphere import NRLMSISE00, SimpleExponentialAtmosphere
from org.orekit.forces.drag import IsotropicDrag, DragForce
from org.orekit.forces.gravity import Relativity
from org.hipparchus.linear import QRDecomposer
from org.hipparchus.optim.nonlinear.vector.leastsquares import GaussNewtonOptimizer
from org.orekit.estimation.leastsquares import BatchLSEstimator
from org.orekit.estimation.iod import IodGooding
from org.orekit.estimation.measurements import AngularAzEl, GroundStation, ObservableSatellite
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.hipparchus.linear import Array2DRowRealMatrix
from org.hipparchus.geometry.euclidean.threed import Vector3D, Rotation

from orekit.pyhelpers import *
from orekit import JArray, JArray_double

#Import Python Libraries
from java.io import File
from datetime import datetime, timedelta
from math import *
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from julian import from_jd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from copy import *

#Import Orbit Determination Libraries
from DataProcessing import *
from Transformations import *
from OrbitDetermination import *
from Variables import *
from InitialOrbitDetermination import *
from StatisticalOrbitDetermination import *

########## DATA MANAGER CONFIGURATION
statusUpdate('Data Manager')

#Setting up Orekit Data Manager
orekit_filename = 'orekit-data'
DM = DataProvidersManager.getInstance()
datafile = File(orekit_filename)
if not datafile.exists():
    print('Directory :', datafile.absolutePath, ' not found')
crawler = DirectoryCrawler(datafile)
DM.clearProviders()
DM.addProvider(crawler)

########## LOCKHEED MARTIN DATA PROCESSING
#statusUpdate('Importing New Data Files')
#initializeDataframeObjects()

########## SATELLITE INFORMATION
statusUpdate('Satellite Information')

#Satellite NORAD ID being observed
noradID = 41851
detectionRequestID = 24809

#Satellite Phyisical Properties
mass, area, cd, cr = cubeSatProperties('3U')

########## INITIAL ORBIT DETERMINATION
statusUpdate('Initial Orbit Determination')


for i in range(0,91):
    scale = 1 - i*0.01
    iod = InitialOrbitDetermination(noradID, detectionRequestID,'gauss',scale)
    report = GenerateIODReport(iod,N41851D24809)
    if scale == 1.0:
        print('TLE Report')
        report.textFileReportTLE()
        print('')
        print('IOD Report')
    report.textFileReportIOD()


input()

########## STATISTICAL ORBIT DETERMINATION
statusUpdate('Statistical Orbit Determination')


#Measurement Parameters
numberOfPasses = 1
measurementSpacing = 5

'''
pod = StatisticalOrbitDetermination(noradID,detectionRequestID,initialOrbit)
pod.defineObjectProperties(mass,area,cd,cr)
finalOrbit = pod.estimate(numberOfPasses,measurementSpacing)

raGuess = pod.raGuess
raMeasure = pod.raMeasure
decGuess = pod.decGuess
decMeasure = pod.decMeasure
time = pod.time
measurementNumber = pod.measurementNumber
trackRMS = pod.trackRMS
trackIter = pod.trackIter
finalEstimate = pod.finalEstimate
'''



def statisticalOrbitDetermination(noradID,detectionRequestID,initialOrbit,numberOfPasses,measurementSpacing):

    #Set up Propagator
    initialEstimate = createPropagator(initialOrbit,mass,area,cd,cr)
    currentEstimate = createPropagator(initialOrbit,mass,area,cd,cr)

    #Initialize Tracking Variables
    trackRMS = []
    trackIter = []
    iteration = 0
    measurement = 0

    #Initialize  RMS values to enter loop
    RMSold = 0
    RMSnew = 1
    dRMS = RMSnew - RMSold

    #Number of Iterations of Batch Least Squares
    while np.linalg.norm(dRMS) >= TOLERANCE and iteration < MAX_ITERATIONS:
    #for k in range(0,1):
        #Initialize Tracking Variables
        raGuess = []
        decGuess = []
        raMeasure = []
        decMeasure = []
        measurementNumber = []
        sumAtWA = []
        sumAtWb = []
        sumb = []
        time = []

        #For each Orbital Pass being processed
        for j in range(0,numberOfPasses):
            #Process Next Orbital Pass
            orbitalPass = Pass(noradID, detectionRequestID[j])
            az, el, ra, dec, datetime, slantRange, rSite = orbitalPass.getConstantSpacing(measurementSpacing)

            #For each measurement in th Orbital Pass
            for i in range(0,len(datetime)):

                #For each dimesion in the State Fector
                for z in range(0,6):
                    #Modify the State for current Dimension
                    currentEstimate, __ = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)
                    modifiedEstimate, dX = modifyPropagator(z,initialEstimate,mass,area,cd,cr)

                    #Propagating State to measurement
                    pvi = currentEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
                    Xi = PVCoordinatesToList(pvi)

                    #Modified State
                    pvMod = modifiedEstimate.propagate(datetime_to_absolutedate(datetime[i])).getPVCoordinates()
                    Xmod = PVCoordinatesToList(pvMod)

                    #Calculate Matricies
                    A = calculateFiniteDifference(Xi,Xmod,dX,datetime[i])
                    b = calculateResiduals(Xi,ra[i],dec[i],datetime[i]) 
                    AtWA, AtWb, btWb = calculateCovarianceMatricies(A,b)

                    #Sum Matricies for each measurement
                    if time == []:
                        sumAtWA = AtWA
                        sumAtWb = AtWb
                        sumbtWb = btWb
                    else:
                        sumAtWA = np.add(sumAtWA,AtWA)
                        sumAtWb = np.add(sumAtWb,AtWb)
                        sumbtWb = np.add(sumbtWb,btWb)

                    #Reset Propagators to initial estimate
                    currentEstimate = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)
                    modifiedEstimate = modifyPropagator('clear',initialEstimate,mass,area,cd,cr)

                #Right Ascension and Declination of current state
                rai, deci = stateVectorToRaDec(Xi,datetime[i])

                #Tracking Variables
                raMeasure.append(ra[i])
                decMeasure.append(dec[i])
                raGuess.append(rai)
                decGuess.append(deci)
                time.append(datetime[i])

        #Average the Root Mean Square of residuals
        btWbAvg = np.divide(sumbtWb,len(time))

        #Inverse of AtWA matrix
        u, s, v = np.linalg.svd(sumAtWA,full_matrices=True)

        #Calculate Covariance Matrix
        P = np.matmul(np.matmul(np.transpose(v),np.linalg.inv(np.diag(s))),np.transpose(u))
        eigval, eigvect = np.linalg.eig(P[0:3,0:3])

        #Calculate Root Mean Square for iteration
        RMS = sqrt(btWbAvg/(2*len(time)))
        
        #Change in Initial State
        dX = np.matmul(P,sumAtWb)
        initialEstimate = updatePropagator(dX,initialEstimate,mass,area,cd,cr)

        #Compare the RMS with previous iteration for exit condition
        RMSold = RMSnew
        RMSnew = RMS
        dRMS = RMSnew - RMSold

        #Counter and Tracking Variables
        iteration += 1
        trackRMS.append(RMS)
        trackIter.append(iteration)

        #Status Update
        if iteration == 10 or iteration == 20 or iteration == 50:
            yn = userInput('Change in RMS is currently '+format(absolute(dRMS), "5.1e")+' at '+str(iteration)+' iterations. Continue? (y/n): ')
            if yn == 'n':
                break
            elif yn == 'y':
                continue 
            else:
                error('invalid option')
        else:
            statusUpdate(iteration)

    measurementNumber = []
    for i in range(0,len(raGuess)):
        measurementNumber.append(i)

    finalOrbit = initialEstimate.getInitialState().getOrbit()

    return raMeasure, raGuess, decMeasure, decGuess, time, trackRMS, trackIter, measurementNumber, initialEstimate, finalOrbit


raMeasure, raGuess, decMeasure, decGuess, time, trackRMS, trackIter, measurementNumber, finalEstimate, finalOrbit = statisticalOrbitDetermination(noradID,detectionRequestID,initialOrbit,numberOfPasses,measurementSpacing)

#raMeasure, raGuess, decMeasure, decGuess, time, trackRMS, trackIter, measurementNumber, initialEstimate, finalOrbit = statisticalOrbitDetermination(noradID,detectionRequestID,finalOrbit,2,measurementSpacing)



########## COMPARING TLE WITH ESTIMATE
statusUpdate('Comparing TLE with Estimate')

#Create TLE Propagator for Satellite
tle = getTLE(noradID,finalOrbit.getDate())
tleEstimate = createSGP4Propagator(tle,mass,area,cd,cr,finalOrbit.getDate())

estimationError = []
timeError = []
for i in range(0,DAY,MINUTE):
    pvTLE = PVCoordinatesToList(tleEstimate.propagate(finalOrbit.getDate().shiftedBy(float(i))).getPVCoordinates())
    pvPOD = PVCoordinatesToList(finalEstimate.propagate(finalOrbit.getDate().shiftedBy(float(i))).getPVCoordinates())

    dX = np.subtract(pvTLE,pvPOD)
    error = np.linalg.norm(dX[0:3])/KM2M

    estimationError.append(error)
    timeError.append(absolutedate_to_datetime(finalOrbit.getDate().shiftedBy(float(i))))


########## PLOTTING RESULTS
statusUpdate('Plotting Results')

#TLE Error Plot
fig1 = make_subplots(rows=1, cols=1)
fig1.add_trace(go.Scatter(x=timeError, y=estimationError), row=1, col=1)
fig1.update_xaxes(title_text="Time", row=1, col=1)
fig1.update_yaxes(title_text="Right Ascension [Degrees]", row=1, col=1)
fig1.update_layout(title='Estimation Error compared to TLE',
                   xaxis_title='Time',
                   yaxis_title='Positional Error [km]')
fig1.show()


#Right Ascension and Declination Error with Iterations
fig2 = make_subplots(rows=1, cols=3,
                    subplot_titles=("Right Ascension", "Declination", "RMS Error"),
                    column_widths=[0.4,0.4,0.2])
fig2.add_trace(go.Scatter(x=measurementNumber, y=raMeasure,
                         mode='markers',
                         legendgroup='group1',
                         name='Measurements',
                         line=dict(color='firebrick', width=4)),
                         row=1, col=1)
fig2.add_trace(go.Scatter(x=measurementNumber, y=raGuess,
                         legendgroup='group1',
                         name='Prediction',
                         line=dict(color='royalblue', width=4)),
                         row=1, col=1)
fig2.update_xaxes(title_text="Time", row=1, col=1)
fig2.update_yaxes(title_text="Right Ascension [Degrees]", row=1, col=1)

fig2.add_trace(go.Scatter(x=measurementNumber, y=decMeasure,
                         legendgroup="group2",
                         name='Measurements',
                         line=dict(color='firebrick', width=4)),
                         row=1, col=2)
fig2.add_trace(go.Scatter(x=measurementNumber, y=decGuess,
                         legendgroup="group2",
                         name='Prediction',
                         line=dict(color='royalblue', width=4)),
                         row=1, col=2)
fig2.update_xaxes(title_text="Time", row=1, col=2)
fig2.update_yaxes(title_text="Declination [Degrees]", row=1, col=2)

fig2.add_trace(go.Scatter(x=trackIter, y=trackRMS,
                         name='RMS Error',
                         line=dict(width=4)),
                         row=1, col=3)
fig2.update_xaxes(title_text="Iterations", row=1, col=3)
fig2.update_yaxes(title_text="RMS Error [Degrees]", row=1, col=3)

fig2.update_layout(title='Comparing Prediction with Measurements',
                   xaxis_title='Time',
                   yaxis_title='Angle (Degrees)')
fig2.show()



