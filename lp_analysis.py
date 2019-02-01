# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 11:26:21 2018

@author: Jia Han

langmuir probe analysis

1. Find linear fitting for Isat and subtract from data
2. Find exponential fitting to Isat
3. Find the crossing point between linear fitting of the transition region and Esat
"""

import math
import numpy
import matplotlib.pyplot as plt
import scipy
from gsmooth import gsmooth


# Probe area
area = 2e-3 * 3e-3 #math.pi * 5.5e-3 * 0.18e-3 #math.pi * (2.5e-3/2)**2 * 2 #double sided planer probe ??? diameter
# electron charge
qe = 1.6e-19 # electron charge (C)
me = 9.11e-31 # electron mass (kg)
mi = 1.67e-27 # proton mass (kg)
k = 1.6e-12 # boltzmann constant (erg/eV)

def analyze_IV(voltage, current, plot = False, value = False):
	"""
	Analyze the IV curve.
	Input:
		voltage: the probe sweep voltage in V
		current: the measured current from the probe in A
		plot: return three plots of the fitting
		value: prints Vp, Te, ne
	Output:
		Vp, Te, ne
	"""

	# Take XX% of the current and fit a linear line
	dif1 = [(max(current) - min(current))*0.05 + min(current), (max(current) - min(current))*0 + min(current)]
	vals = numpy.argwhere(numpy.logical_and(current < dif1[0], current > dif1[1]))

	cropped_voltage = []
	cropped_current = []
	for i in range(0, len(vals)):
		idx = vals[i][0]
		cropped_voltage.append(voltage[idx])
		cropped_current.append(current[idx])

	c = numpy.polyfit(cropped_voltage, cropped_current, 1)
	y = c[0] * voltage + c[1]
	
	print('Isat linear parameter:', c[0])

	if plot:
		plt.figure(1)
		plt.plot(voltage,current)
		plt.plot(voltage,y)

	Inew = current - y
	Vnew = voltage


  # Take XX% of the remaining current and perform exponential fitting
	dif2 = [min(Vnew) + (max(Vnew) - min(Vnew))*0.6, (max(Inew) - min(Inew))*0.1]
	vals_new = numpy.argwhere(numpy.logical_and(Vnew > dif2[0], Inew < dif2[1]))
		
	Vnew_cropped = Vnew[vals_new[0][0]:vals_new[-1][0]]
	Inew_cropped = Inew[vals_new[0][0]:vals_new[-1][0]]

	a, b = scipy.optimize.curve_fit(lambda t, a, b: a*numpy.exp(b*t), Vnew_cropped, Inew_cropped)

	def f(x):
		return a[0] * numpy.exp(a[1] * x)


	if plot:
		plt.figure(2)
		plt.plot(Vnew, Inew)
		plt.scatter(Vnew_cropped,Inew_cropped, color='r', label='Data points that were fitted')
#		plt.scatter(Vnew_cropped, gen_I, color='r')
		plt.plot(Vnew_cropped, f(numpy.asarray(Vnew_cropped)))
		
		plt.legend()

	Te = 1/a[1]
	if value:
		print ("Te =%.2f"%(Te), "eV")
	
	if Te > 10:
		raise Exception('Te is very high')


	# Defines which region is the transition
	dif3 = (max(Inew) - min(Inew))* 5/10 + min(Inew) # Upper limit
	dif4 = (max(Inew) - min(Inew))*1/5 + min(Inew) # Lower limit

	lower_bound = numpy.argwhere(Inew > dif4)
	start_idx = lower_bound[0][0]
	upper_bound = numpy.argwhere(Inew < dif3)
	stop_idx = upper_bound[len(upper_bound)-1][0]

	trans_voltage = []
	trans_current = []
	for i in range(start_idx, stop_idx):
		trans_voltage.append(Vnew[i])
		trans_current.append(Inew[i])
	c = numpy.polyfit(trans_voltage, trans_current, 1)
	y = c[0] * Vnew + c[1]


  # Finds linear fitting to Esat
	dif5 = min(Inew) + (max(Inew) - min(Inew)) * 0.8
	esat_pos = numpy.argwhere(Inew > dif5)

	esat_volt = []
	esat_curr = []

	for i in esat_pos[:,0]:
		esat_volt.append(Vnew[i])
		esat_curr.append(Inew[i])
	d = numpy.polyfit(esat_volt, esat_curr, 1)
	z = d[0] * Vnew + d[1]

	if plot:
		plt.figure(3)
		plt.plot(Vnew, Inew)
		plt.plot(Vnew,y)
		plt.plot(esat_volt,esat_curr)
		plt.plot(Vnew, z)

	# Find the crossing point of transition and Esat linear fit to produce ne and Vp
	Vp = abs((d[1]-c[1]) / (d[0] - c[0])) #plasma potential in V
	I = d[0] * Vp + d[1]                  #electron current in A


	if Te > 0:
		vth = math.sqrt(qe*Te / me)
								# electron thermal velocity in cm/s
		ne = I/(vth * area* qe)
	else:
		ne = 0
		raise Exception('Te is negative')

	if value:
		print ('Esat=%.2g'%(I/area*1e-4), 'A/cm^2')
		print ('ne=%.2g'%(ne), 'm^3')
		print ('Plasma potential=%.2f'%(Vp), 'V \n')

	return (Vp, Te, ne)
	
	
#########################################################################################################
'''
Not used anymore
'''

r1 = 200e3 # Measure resistor
r2 = 40e3 # Sweep resistor

def IV_curve(Vsweep, Vmeas, tpos, plot=False):
    
    V_probe = gsmooth(Vmeas,50) * (r1+50)/50
    voltage = gsmooth(V_probe, 50)
    current = (Vsweep - V_probe)/r2 - V_probe/r1
    
    zero_cross_arr, period = general.zerocrossing_avg_period(Vsweep, 1)
    start = zero_cross_arr[tpos] - int(period/4) + 90
    stop = zero_cross_arr[tpos+1] - int(period/4)- 90
    
    if plot:
        plt.figure()
        plt.plot(Vsweep[start:stop])
        plt.figure()
        plt.plot(Vmeas[start:stop])
        
        print (start, stop)
    
    V_rise = voltage[start:start+int(period/2)]
    V_fall = voltage[stop-int(period/2):stop]
    I_rise = current[start:start+int(period/2)]
    I_fall = current[stop-int(period/2):stop]

    V_avg = (V_rise + numpy.flip(V_fall,0))/2
    I_avg = (I_rise + numpy.flip(I_fall,0))/2
    
    if plot:
        plt.figure()
        plt.plot(V_rise, I_rise)
        plt.plot(V_fall, I_fall)
        plt.plot(V_avg, I_avg)
        
    return V_avg, I_avg