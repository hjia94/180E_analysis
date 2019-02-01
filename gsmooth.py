# -*- coding: utf-8 -*-
"""
	Perform smoothing on an input data set, by convolving with a Gaussian kernel.

	FWIW, this is *way* faster than scipy.ndimage.filters.gaussian_filter for large data sets,
		which actually seems to be using an O(N**2) algorithm   (2017-07-29)

@file:   gsmooth.py
@author: Patrick
@date:   Created: 2015-09-27
	      Updated: 2017-07-29: fix to use scipy.signal.gaussian rather than home-brew DLL for kernel
	      Last update: 2018-04-28: fix kernel area normalization - previously for _some_ intervals the filtered data amplitude was changed by a few percent (e.g. interval = 24, for 10002 points)
"""
import numpy
import scipy.signal
import scipy.special
import math

def gsmooth(data, interval):
	""" Return smoothed version of the data via convolution with a Gaussian of width interval #points.
		For a step (example below) the output approaches ~ 0.92 of its final value at 1 interval after the rise.
		Here the kernel is A * exp(-i**2/interval**2), where A is calculated to give area = 1 (uses scipy erf()).
		bug: cases where interval is very wide and the edges are cut off get funny looking. because of this the
		     erf() correction is pretty superfluous
	"""
	n = int(interval/math.sqrt(2))
	kernel = scipy.signal.gaussian(data.size, n)
	interval = n * math.sqrt(2)  # need to adjust interval to match the kernel!!!  (added 2018-04-28)
	kernel /= interval * math.sqrt(math.pi) * scipy.special.erf(data.size/interval)   # fix for area

	# note: mode=
	#			'full'  - The output is the full discrete linear convolution of the inputs. (Default)
	#			'valid' - The output consists only of those elements that do not rely on the zero-padding.
	#			'same'  - The output is the same size as data, centered with respect to the ‘full’ output.
	#    The first two do not work in this application
	return scipy.signal.fftconvolve(kernel, data, mode='same')	  # use mode 'same'; boundary effects are still visible.

#===============================================================================================================================================
#<o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o> <o>
#===============================================================================================================================================

if __name__ == '__main__':
	import pylab as plt
	import scipy.ndimage
	import time

	num_pts  = 100000
	interval =   5000

	print('generating step data (%i points)'%num_pts, flush=True)
	data = numpy.zeros(shape=(num_pts), dtype=numpy.float)
	data[int(num_pts/2):] = 1

	plt.plot(data)

	print('filtering (interval = ', interval,')', sep='', flush=True)
	t0 = time.time()

	sdata = gsmooth(data, interval)

	t1 = time.time()
	print("  %.3f sec"%(t1-t0), flush=True)

	plt.plot(sdata)

	if True:
		print('filtering using scipy.ndimage.filters.gaussian_filter', flush=True)
		t0 = time.time()
		rdata = scipy.ndimage.filters.gaussian_filter1d(data, interval)
		t1 = time.time()
		print("  %.3f sec"%(t1-t0), flush=True)
		plt.plot(rdata)

	plt.show()
