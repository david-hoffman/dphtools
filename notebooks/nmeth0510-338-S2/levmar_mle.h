/* 
////////////////////////////////////////////////////////////////////////////////////
// 
//  Prototypes and definitions for the Levenberg - Marquardt minimization algorithm
//  Copyright (C) 2004  Manolis Lourakis (lourakis at ics forth gr)
//  Institute of Computer Science, Foundation for Research & Technology - Hellas
//  Heraklion, Crete, Greece.
//
//  Modifications Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at the Lawrence Livermore National Laboratory. Written by Ted Laurence (laurence2@llnl.gov)
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
////////////////////////////////////////////////////////////////////////////////////
*/

#include <float.h>
#ifndef _LEVMAR_H_
#define _LEVMAR_H_


#ifdef __cplusplus
extern "C" {
#endif


#define FABS(x) (((x)>=0.0)? (x) : -(x))

/* work arrays size for ?levmar_der and ?levmar_dif functions.
 * should be multiplied by sizeof(double) or sizeof(float) to be converted to bytes
 */
#define LM_MLE_WORKSZ(npar, nmeas) (3*(nmeas) + 4*(npar) + (nmeas)*(npar) + 2*(npar)*(npar))

#define LM_OPTS_SZ    	 5 
#define LM_INFO_SZ    	 10
#define LM_ERROR         -1
#define LM_INIT_MU    	 1E-03
#define LM_STOP_THRESH	 1E-20
#define LM_DIFF_DELTA    1E-20
#define LM_VERSION       "2.5 (December 2009)"

#define LM_CHISQ_MLE		0
#define LM_CHISQ_NEYMAN		1
#define LM_CHISQ_EQUAL_WT	2


/* double precision LM, with Jacobian */
/* unconstrained minimization */
extern int dlevmar_mle_der(
      void (*func)(double *p, double *hx, int m, int n, void *adata),
      void (*jacf)(double *p, double *j, int m, int n, void *adata),
      double *p, double *x, int m, int n, int itmax, double *opts,
      double *info, double *work, double *covar, void *adata, int fitType);

extern double compute_chisq_measure(double *e, double *x, double *hx, int n, int fitType);



#ifdef __cplusplus
}
#endif

#endif /* _LEVMAR_H_ */
