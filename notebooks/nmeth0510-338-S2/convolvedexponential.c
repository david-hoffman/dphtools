/* convolvedexponential.c -- model functions for multiple exponentials + background, convolved with an instrument response */
/////////////////////////////////////////////////////////////////////////////////
// 
//  Calculation of multiple exponentials convolved with an instrument response.  
//  using GNU Scientific Library.  Also, calls Levenberg-Marquardt minimization routine
//  for fitting event counting histograms.  Options include using the Maximum Likelihood Estimator
//  for Poisson deviates and least squares fitting.
//
//  Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory. Written by Ted Laurence (laurence2@llnl.gov)
//  LLNL-CODE-424602 All rights reserved.
//  This file is part of dlevmar_mle_der
//
//  Please also read Our Notice and GNU General Public License.
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License 
//  along with this program; if not, write to the 
//  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
//
/////////////////////////////////////////////////////////////////////////////////
 
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_vector.h>
#include "levmar_mle.h"


struct data {
	double deltaT;	/*Spacing between bins*/
	size_t n;		/*Number of data elements*/
	size_t nRadix2; /*Next higher power of 2 for n, giving size of irf and fWork arrays*/
	double *irfWork;/*Instrument Response Function*/
	double *fWork;  /*Workspace for function calculation*/
	size_t m_total;	/*Number of parameters (number of exponentials = (m-1)/2 ) */
	double *a;		/*Parameter Array; fitted parameters x are subsituted into this array */
	size_t m;		/*Number of fitted parameters*/
	int *ip;		/*Mapping of fitted parameter array x to parameter array a*/
};

void complex_convolution_multiply(double *a, const double *b, int n)
{
	int i,j;
	double re,im,factor;

	factor = 1.0/n; /* to normalize inverse FFT */

	a[0]*=b[0]*factor;
	a[n/2]*=b[n/2]*factor;

	for (i=1;i<n/2;i++)
	{
		j=n-i;
		re=a[i];
		im=a[j];
		a[i]=(re*b[i]-im*b[j])*factor;
		a[j]=(im*b[i]+re*b[j])*factor;
	}
}

void convolved_exp_f(double *p, double *f, int m, int n, void *data)
{
	double deltaT=((struct data *)data)->deltaT;
	size_t nRadix2 = ((struct data *)data)->nRadix2;
	double *irfWork = ((struct data *) data)->irfWork;
	double *fWork = ((struct data *) data)->fWork;
	size_t m_total = ((struct data *)data)->m_total;
	double *a = ((struct data *) data)->a;
	int *ip = ((struct data *) data)->ip;

	int i,j;
	int numExponentials = (m_total-1)/2;
	double tau, amp, sqrtTau,sqrtAmp;

	double constantValue;
	double totalT=deltaT*n;
	double factor1;
	double factor2;

	if ( (n != ((struct data *)data)->n ) || (m != ((struct data *)data)->m) ) return;


	for (i=0; i<m; i++)
		a[ ip[i] ] = p[i];

	constantValue=a[0]*a[0]/n;

	for (i=0; i<n; i++) fWork[i] =  constantValue;
	for (i=n; i<nRadix2; i++) fWork[i] = 0.0;


	for (j=0;j<numExponentials;j++)
	{
		sqrtAmp=a[2*j+1];
		sqrtTau=a[2*j+2];
		amp=sqrtAmp*sqrtAmp;
		tau=sqrtTau*sqrtTau;
		factor1=1-exp(-deltaT/tau);
		factor2=1-exp(-totalT/tau);
		for (i=0;i<n;i++)
			fWork[i]+=amp*factor1/factor2*exp(-deltaT*i/tau);
	}

	gsl_fft_real_radix2_transform(fWork,1,nRadix2);
	complex_convolution_multiply(fWork,irfWork,nRadix2);
	gsl_fft_halfcomplex_radix2_backward(fWork,1,nRadix2);

	// This loop uses 0 offset indexing
	for (i=n;i<nRadix2;i++) fWork[i%n]+=fWork[i];
	for (i=0;i<n;i++) f[i]=fWork[i];

	return;
}

void convolved_exp_df(double *p, double *jac, int m, int n, void *data)
{
	double deltaT=((struct data *)data)->deltaT;
	size_t nRadix2 = ((struct data *)data)->nRadix2;
	double *irfWork = ((struct data *) data)->irfWork;
	double *fWork = ((struct data *) data)->fWork;
	size_t m_total = ((struct data *)data)->m_total;
	double *a = ((struct data *) data)->a;
	int *ip = ((struct data *) data)->ip;
	double *jac_row;

	int i,k;

	if ( (n != ((struct data *)data)->n ) || (m != ((struct data *)data)->m) ) return;

	for (k=0; k<m; k++)
	{
		double tau, amp, sqrtTau,sqrtAmp;
		double temp, totalT=deltaT*n;
		double factor1;
		double factor2;

		if (ip[k]==0)
		{
			double d_constant=2*a[0]/n;
			jac_row=jac;
			for (i=0;i<n;i++,jac_row+=m)
				jac_row[k]=d_constant; 
			continue;
		}

		for (i=0; i<nRadix2; i++) fWork[i] = 0.0;

		if ((ip[k]-1)%2==0) /* Amplitude */
		{
			sqrtAmp=a[ip[k]];
			sqrtTau=a[ip[k]+1];
			amp=sqrtAmp*sqrtAmp;
			tau=sqrtTau*sqrtTau;
			factor1=1-exp(-deltaT/tau);
			factor2=1-exp(-totalT/tau);
			for (i=0;i<n;i++)
				fWork[i]=2*sqrtAmp*factor1/factor2*exp(-deltaT*i/tau);
		}
		else /* lifetime */
		{
			sqrtAmp=a[ip[k]-1];
			sqrtTau=a[ip[k]];
			amp=sqrtAmp*sqrtAmp;
			tau=sqrtTau*sqrtTau;
			factor1=1-exp(-deltaT/tau);
			factor2=1-exp(-totalT/tau);
			for (i=0;i<n;i++)
			{
				temp=factor1/factor2*exp(-deltaT*i/tau);
				fWork[i]= -2*amp*deltaT*exp(-(deltaT*i+deltaT)/tau)/(factor2*tau*sqrtTau)
							+2*temp*amp*deltaT*i/(tau*sqrtTau)
							+2*temp*amp*totalT*exp(-totalT/tau)/(factor2*tau*sqrtTau);
			}
		}

		gsl_fft_real_radix2_transform(fWork,1,nRadix2);
		complex_convolution_multiply(fWork,irfWork,nRadix2);
		gsl_fft_halfcomplex_radix2_backward(fWork,1,nRadix2);

		// This loop uses 0 offset indexing
		for (i=n;i<nRadix2;i++) fWork[i%n]+=fWork[i];

		jac_row=jac;
		for (i=0;i<n;i++,jac_row+=m)
			jac_row[k]=fWork[i]; 
	}
	return;
}

int fit_convolved_exponential(int n,			// number of measurement bins
							  double deltaT,	// time resolution for each bin
							  double *x,		// Data Measured
							  double *f,		// Space for function
							  double *irf,		// Instrument Response
							int m_total,		// Total number of parameters (including those not fitted)
							double *a,			// Parameter array (length m_total)
							int m,				// Number of fitted parameters
							int *ip,			// Array of length m with indices of fitted parameters in a
							double *chisq,		// Result for chisq
							int *nIterations,	// Number of iterations performed
							int fitType,		// 0 = MLE, 1 = Neyman weighting, 2 = Equal Weighting (sigma=1)
							double deltaChisqLimit, // Stop criterion for change in chisq
							int maxIterations)	//  maximum number of iterations
{
	struct data importedData;

	double tempN,nRadix2;
	double *irfWork, *fWork;
	double opts[LM_OPTS_SZ],info[LM_INFO_SZ];

	int i;
	int status;
	unsigned iter = 0;

	double *p;

	opts[0]=LM_INIT_MU;
	opts[1]=LM_STOP_THRESH;
	opts[2]=LM_STOP_THRESH;
	opts[3]=LM_STOP_THRESH;
	opts[4]=deltaChisqLimit;

	importedData.n=n;
	importedData.deltaT=deltaT;
	importedData.m_total=m_total;
	importedData.a=a;
	importedData.m=m;
	importedData.ip=ip;

	for (i=0; i<m_total; i++) a[i]=sqrt(a[i]);

	/* Determine power of 2 higher than n required for performing convolutions via FFT */
	tempN = pow(2, 1.0+ceil( log((double)n)/log(2.0) ) );
	nRadix2 = ( (int) tempN );
	irfWork=malloc(nRadix2*sizeof(double));
	if (irfWork==NULL) return GSL_ENOMEM;
	fWork=malloc(nRadix2*sizeof(double));
	if (fWork==NULL) 
	{
		free(irfWork);
		return GSL_ENOMEM;
	}
	p=malloc(m*sizeof(double));
	if (p==NULL) 
	{
		free(irfWork);
		free(fWork);
		return GSL_ENOMEM;
	}

	for (i=0; i<n; i++)
		irfWork[i]=irf[i];
	for (i=n; i<nRadix2; i++)
		irfWork[i]=0.0;
	gsl_fft_real_radix2_transform(irfWork,1,nRadix2);

	importedData.nRadix2=nRadix2;
	importedData.irfWork=irfWork;
	importedData.fWork=fWork;

	for (i=0; i<m; i++)
		p[i]=a[ip[i]];

	status = dlevmar_mle_der( &convolved_exp_f, &convolved_exp_df, p, x, m, n, maxIterations,opts,info,NULL,NULL,&importedData,fitType);

	*chisq = info[1]/(n-m);
	*nIterations=info[5];

	convolved_exp_f(p, f, m, n, &importedData);  /* Evaluate function for external use in array f */

	for (i=0; i<m; i++)
		a[ip[i]]=p[i];

	free(p);
	free(fWork);
	free(irfWork);

	for (i=0; i<m_total; i++) a[i]=a[i]*a[i];

	return info[6];
}





