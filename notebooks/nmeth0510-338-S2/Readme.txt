Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory. Written by Ted Laurence (laurence2@llnl.gov)
LLNL-CODE-424602 All rights reserved.
This file is part of dlevmar_mle_der

Please also read Our Notice and GNU General Public License.
This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License (as published by the Free Software Foundation) version 2, dated June 1991.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA


--------------

In the these C code files, we implemented a version 
of the Levenberg-Marquardt algorithm for fitting a 
sum of exponentials convolved with an instrument 
response function (IRF).  

The file “convolvedexponential.c” contains functions 
for calculating the model as well as the entry point 
for calling fitting routines from external software.  
Note that we use squared parameters to enforce 
positivity.  The main functions are:

fit_convolved_exponential 	entry point for fitting data
convolved_exp_f 		calculates function
convolved_exp_df		calculates Jacobian

These routines must be called by external software since
there is no main() function.  We implemented our fitting
routine as a DLL on a PC, and called the routine from 
National Instrument LabVIEW, although there is no reason 
the routines cannot be used with other compilers.

The function "fit_convolved_exponential" allows the user
to select which parameters in the function will be fitted. 
Each element in the array "ip" is the index of the elements
of the parameter array "a" that will be fitted.  

The parameters in the array "a" are:
a[0] = background constant
a[1] = amplitude of first lifetime
a[2] = first lifetime
a[3] = amplitude of second lifetime
a[4] = second lifetime
...
a[2*n-1] = amplitude of nth lifetime
a[2*n] = nth lifetime


The parameters for the function "fit_convolved_exponential"
are the following:

int n,			// number of measurement bins
double deltaT,		// time resolution for each bin
double *x,		// Data Measured (size n array)
double *f,		// Space for function (size n array)
double *irf,		// Instrument Response (size n array)
int m_total,		// Total number of parameters (including those not fitted)
double *a,		// Parameter array (length m_total)
int m,			// Number of fitted parameters
int *ip,		// Array of length m with indices of fitted parameters in a
double *chisq,		// Result for chisq
int *nIterations,	// Number of iterations performed
int fitType,		// 0 = MLE, 1 = Neyman weighting least squares, 2 = Equal Weighting least squares(sigma=1)
double deltaChisqLimit, // Stop criterion for change in chisq
int maxIterations	//  maximum number of iterations

The actual minimization routine dlevmar_mle_der is contained 
in file “lm_mle.c”, and requires the two headers 
“lm_mle_compiler.h” and “levmar_mle.h”.  These files are 
significantly modified forms of the code in the levmar package 
(http://www.ics.forth.gr/~lourakis/levmar/).  In the levmar 
routines, there are many options for single or double 
precision floating point arithmetic and constraints.  
For our purposes, those were unnecessary complications.  
We therefore stripped the code of all of those options, 
and simply have a double precision implementation without 
fitting constraints.  The main differences in the code 
for use with the MLE vs. least squares can be found in 
the switch statements in the “lm_mle.c” file. 


We use the GNU Scientific Library (GSL) routines for linear 
algebra in the L-M routine, and for the FFT routine 
for convolution calculations in accounting for 
instrument response (http://www.gnu.org/software/gsl/).  
For this reason, these routines must be linked with the 
GNU scientific library before use.

We used Microsoft Visual Studio, and linked these routines
to the port of GSL 1.13 provided by David Geldreich at
http://david.geldreich.free.fr/dev.html

