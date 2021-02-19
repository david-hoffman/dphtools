#!/usr/bin/env python
# -*- coding: utf-8 -*-
# lpsvd.py
"""
Conversion of old IgorPro code.

LPSVD was developed by Tufts and Kumaresan (Tufts, D.; Kumaresan, R. IEEE Transactions on Acoustics,
Speech and signal Processing 1982, 30, 671 – 675.) as a method of harmonic inversion, i.e. decomposing
a time signal into a linear combination of (decaying) sinusoids.

A great reference that is easy to read for the non-EECS user is:
Barkhuijsen, H.; De Beer, R.; Bovée, W. M. M. .; Van Ormondt, D. J. Magn. Reson. (1969) 1985, 61, 465–481.

This particular implementation was adapted, in part, from matNMR by Jacco van Beek
http://matnmr.sourceforge.net/
and  Complex Exponential Analysis by Greg Reynolds
http://www.mathworks.com/matlabcentral/fileexchange/12439-complex-exponential-analysis/

Author: David Hoffman (dave.p.hoffman@gmail.com)
Date: Aug, 2015
"""

# import numpy as np
# import pandas as pd
# from scipy.linalg import hankel, inv, pinv, svd

# # K_SPEED_OF_LIGHT = 2.99792458e-5 #(cm/fs)


# def LPSVD(signal, M=None, lfactor=1 / 2, removebias=True):
#     """
#     A function that performs the linear prediction-singular value decomposition
#     of a signal that is assumed to be a linear combination of damped sinusoids

#     Parameters
#     ----------
#     signal : ndarray
#         The signal to be analyzed
#     M : int
#         Model order, if None, it will be estimated
#     lfactor : float
#         How to size the Hankel matrix, Tufts and Kumaresan suggest 1/3-1/2
#         Default number of prediction coefficients is half the number of points
#         in the input wave
#     removebias    : bool
#         If true bias will be removed from the singular values of A

#     """
#     if lfactor > 3 / 4:
#         print("You attempted to use an lfactor greater than 3/4, it has been set to 3/4")
#         lfactor = 3 / 4

#     # length of signal
#     N = len(signal)
#     # Sizing of the Hankel matrix, i.e. the backward prediction matrix
#     L = int(np.floor(N * lfactor))
#     # Shift the signal forward by 1
#     rollsig = np.roll(signal, -1)
#     # Generate the Hankel matrix
#     A = hankel(rollsig[: N - L], signal[L:])
#     # Take the conjugate of the Hankel Matrix to form the prediction matrix
#     A = np.conj(A)
#     # Set up the data vector, the vector to be "predicted"
#     h = signal[: N - L]
#     h = np.conj(h)  # Take the conjugate

#     U, S, VT = svd(A)  # Perform an SVD on the Hankel Matrix

#     # We can estimate the model order if the user hasn't selected one
#     if M is None:
#         M = estimate_model_order(S, N, L) + 8
#         print("Estimated model order: {}".format(M))

#     if M > len(S):
#         M = len(S)
#         print("M too large, set to max = ".format(M))

#     # remove bias if needed
#     if removebias:
#         # Here we subtract the arithmatic mean of the singular values determined to be
#         # noise from the rest of the singular values as described in Barkhuijsen
#         S -= S[M:].mean()

#     S = 1 / S[:M]  # invert S and truncate

#     # Redimension the matrices to speed up the matrix multiplication step
#     VT = VT[:M, :]  # Make VT the "right" size
#     U = U[:, :M]  # Make U the "right" size

#     # Now we can generate the LP coefficients
#     lp_coefs = -1 * np.conj(VT.T).dot(np.diag(S)).dot(np.conj(U.T)).dot(h)

#     # Error check: are there any NaNs or INFs in lp_coefs?
#     if not np.isfinite(lp_coefs).all():
#         raise ValueError(
#             "There has been an error generating the prediction-error filter polynomial"
#         )

#     # Need to add 1 to the beginning of lp_coefs before taking roots
#     lp_coefs = np.insert(lp_coefs, 0, 1)

#     # I can now find the roots of B (assuming B represents the coefficients of a polynomial)
#     # Note that NumPy defines polynomial coefficients with the larges power first
#     # so we have to reverse the coefficients before finding the roots.
#     myroots = np.roots(lp_coefs[::-1])

#     # Remove the poles that lie within the unit circle on the complex plane as directed by Kurmaresan
#     # Actually it seems the correct thing to do is to remove roots with positive damping constants
#     usedroots = np.array([np.conj(np.log(root)) for root in myroots if np.abs(root) <= 1])

#     # Error checking: see if we removed all roots!
#     if len(usedroots) == 0:
#         raise ValueError("There has been an error finding the real poles")

#     # sort by freqs
#     usedroots = usedroots[np.imag(usedroots).argsort()]
#     # Lets make a DataFrame with dimension labels to store all our parameters
#     lpsvd_coefs = pd.DataFrame(columns=["amps", "freqs", "damps", "phase"])

#     # We can directly convert our poles into estimated damping factors and frequencies
#     lpsvd_coefs['damps'] = np.real(usedroots)
#     lpsvd_coefs['freqs'] = np.imag(usedroots) / (2 * np.pi)

#     # But we need to do a little more work to get the predicted amplitudes and phases
#     # Here we generate our basis matrix
#     basis = np.array([np.exp(np.arange(len(signal)) * root) for root in usedroots])

#     # Take the inverse
#     pinvBasis = pinv(basis)

#     # And apply it to our signal to recover our predicted amplitudes
#     # Amps here are complex meaning it has amplitude and phase information
#     cAmps = pinvBasis.T.dot(signal)

#     lpsvd_coefs.amps = np.abs(cAmps)
#     lpsvd_coefs.phase = np.angle(cAmps)

#     # Calculate the errors
#     calc_LPSVD_error(lpsvd_coefs, signal)

#     return lpsvd_coefs  # , Errors


# def estimate_model_order(s, N, L):
#     """
#     Adapted from from Complex Exponential Analysis by Greg Reynolds
#     http://www.mathworks.com/matlabcentral/fileexchange/12439-complex-exponential-analysis/
#     Use the MDL method as in Lin (1997) to compute the model
#     order for the signal. You must pass the vector of
#     singular values, i.e. the result of svd(T) and
#     N and L. This method is best explained by Scharf (1992).

#     Parameters
#     ----------
#     s : ndarray
#         singular values from SVD decomposition
#     N : int
#     L : int

#     Returns
#     -------
#     M : float
#         Estimated model order
#     """
#     MDL = np.zeros(L)

#     for i in range(L):
#         MDL[i] = -N * np.log(s[i:L]).sum()
#         MDL[i] += N * (L - i) * np.log(s[i:L].sum() / (L - i))
#         MDL[i] += i * (2 * L - i) * np.log(N) / 2

#     return MDL.argmin()


# def calc_LPSVD_error(LPSVD_coefs, data):
#     """
#     A function that estimates the errors on the LPSVD parameters using the Cramer-Rao
#     lower bound (http://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93Rao_bound).
#     This implementation is based on the work of Barkhuijsen et al (http://dx.doi.org/10.1016/0022-2364(86)90446-4)

#     Parameters
#     ----------
#     LPSVD_coefs    : DataFrame
#         Coefficients calculated from the LPSVD algorithm, we will add errors to this DataFrame
#     data : ndarray
#         The data from which the LPSVD coefficients were calculated
#     """

#     # ***The first thing to do is to calculated the RMS of the residuals***
#     # We reconstruct the model from the parameters
#     recon = reconstruct_signal(LPSVD_coefs, data)
#     p = np.arange(len(data))

#     res = data - recon

#     # Calculate the RMS
#     RMS = np.sqrt((res ** 2).mean())

#     # Next we need to generate the Fisher matrix
#     size = len(LPSVD_coefs) * 4
#     FisherMat = np.zeros((size, size))

#     # We'll reuse res for the intermediate calculations
#     # This implementation is based on the work of Barkhuijsen et al (http://dx.doi.org/10.1016/0022-2364(86)90446-4)
#     for i, rowi in LPSVD_coefs.iterrows():
#         ampi = rowi.amps
#         freqi = rowi.freqs
#         dampi = rowi.damps
#         phasei = rowi.phase
#         for j, rowj in LPSVD_coefs.iterrows():
#             ampj = rowj.amps
#             freqj = rowj.freqs
#             dampj = rowj.damps
#             phasej = rowj.phase

#             res = np.exp(
#                 p * complex(dampi + dampj, 2 * np.pi * (freqi - freqj))
#                 + complex(0, 1) * (phasei - phasej)
#             )
#             chi0 = np.real(res).sum()
#             zeta0 = np.imag(res).sum()

#             res = p * np.exp(
#                 p * complex(dampi + dampj, 2 * np.pi * (freqi - freqj))
#                 + complex(0, 1) * (phasei - phasej)
#             )
#             chi1 = np.real(res).sum()
#             zeta1 = np.imag(res).sum()

#             res = p ** 2 * np.exp(
#                 p * complex(dampi + dampj, 2 * np.pi * (freqi - freqj))
#                 + complex(0, 1) * (phasei - phasej)
#             )
#             chi2 = np.real(res).sum()
#             zeta2 = np.imag(res).sum()

#             # First Row
#             FisherMat[4 * i + 0][4 * j + 0] = ampi * ampj * chi2
#             FisherMat[4 * i + 0][4 * j + 1] = -ampi * zeta1
#             FisherMat[4 * i + 0][4 * j + 2] = ampi * ampj * zeta2
#             FisherMat[4 * i + 0][4 * j + 3] = ampi * ampj * chi1
#             # Second Row
#             FisherMat[4 * i + 1][4 * j + 0] = ampj * zeta1
#             FisherMat[4 * i + 1][4 * j + 1] = chi0
#             FisherMat[4 * i + 1][4 * j + 2] = -ampj * chi1
#             FisherMat[4 * i + 1][4 * j + 3] = ampj * zeta0
#             # Third Row
#             FisherMat[4 * i + 2][4 * j + 0] = -ampi * ampj * zeta2
#             FisherMat[4 * i + 2][4 * j + 1] = -ampi * chi1
#             FisherMat[4 * i + 2][4 * j + 2] = ampi * ampj * chi2
#             FisherMat[4 * i + 2][4 * j + 3] = -ampi * ampj * zeta1
#             # Fourth Row
#             FisherMat[4 * i + 3][4 * j + 0] = ampi * ampj * chi1
#             FisherMat[4 * i + 3][4 * j + 1] = -ampi * zeta0
#             FisherMat[4 * i + 3][4 * j + 2] = ampi * ampj * zeta1
#             FisherMat[4 * i + 3][4 * j + 3] = ampi * ampj * chi0

#     FisherMat = inv(FisherMat)  # Replace the Fisher matrix with its inverse
#     FisherMat *= 2 * RMS ** 2

#     LPSVD_coefs.insert(4, "amps_error", np.nan)
#     LPSVD_coefs.insert(5, "freqs_error", np.nan)
#     LPSVD_coefs.insert(6, "damps_error", np.nan)
#     LPSVD_coefs.insert(7, "phase_error", np.nan)
#     # Fill up the Error wave with the errors.
#     for i in range(len(LPSVD_coefs)):
#         LPSVD_coefs.amps_error.loc[i] = np.sqrt((FisherMat[1 + i * 4][1 + i * 4]))
#         LPSVD_coefs.freqs_error.loc[i] = np.sqrt((FisherMat[0 + i * 4][0 + i * 4]))
#         LPSVD_coefs.damps_error.loc[i] = np.sqrt((FisherMat[2 + i * 4][2 + i * 4]))
#         LPSVD_coefs.phase_error.loc[i] = np.sqrt((FisherMat[3 + i * 4][3 + i * 4]))

#     return LPSVD_coefs


# def reconstruct_signal(LPSVD_coefs, signal, ampcutoff=0, freqcutoff=0, dampcutoff=0):
#     """
#     #A function that reconstructs the original signal in the time domain and frequency domain
#     #from the LPSVD algorithms coefficients, which are passed as LPSVD_coefs
#     #http://mathworld.wolfram.com/FourierTransformLorentzianFunction.html

#     WAVE LPSVD_coefs        #coefficients from the LPSVD algorithm
#     String name                #Name of the generated waves
#     Variable length            #Length of the time domain signal
#     Variable timeStep        #Sampling frequency with which the signal was recorded, in fs
#     Variable dataReal        #Should the output time domain data be real?
#     Variable ampcutoff        #Cutoff for the amplitudes of the components
#     Variable freqcutoff        #Cutoff for the frequency of the components
#     Variable dampcutoff        #Cutoff for the damping of the components
#     """

#     # Initialize time domain signal
#     time_domain = np.zeros_like(signal, dtype=complex)
#     p = np.arange(len(signal))

#     for i, row in LPSVD_coefs.iterrows():
#         damp = -row.damps / np.pi
#         if row.amps ** 2 > ampcutoff and damp >= dampcutoff:
#             # Keep in mind that LPSVD_coefs were constructed agnostic to the actual sampling
#             # frequency so we will reconstruct it in the same way
#             amp = row.amps
#             damp = row.damps
#             phase = row.phase
#             freq = row.freqs
#             time_domain += amp * np.exp(
#                 p * complex(damp, 2 * np.pi * freq) + complex(0, 1) * phase
#             )

#     if signal.dtype != complex:
#         time_domain = np.real(time_domain)

#     return time_domain

# ### NOTE: old IgorPro code below

# # Function/S Cadzow(signal, M, iters,[lfactor,q])
# #     #Remove noise using the Cadzow composite property mapping method.
# #     #See Cadzow, J. A. IEEE Transactions on Acoustics, Speech and signal Processing 1988, 36, 49 –62.
# #     #Adapted from from Complex Exponential Analysis by Greg Reynolds
# #     #http://www.mathworks.com/matlabcentral/fileexchange/12439-complex-exponential-analysis/
# #
# #     Wave signal        #The signal to be filtered
# #     Variable M        #The expected number of signals (2 times the number of damped sinusoids
# #     Variable iters    #Number of iterations to be performed
# #
# #     Variable lfactor    #User selectable factorization of the Hankel Matrix
# #     Variable q        #Verbose or not
# #
# #     if(ParamIsDefault(lfactor))
# #         lfactor = 1/2
# #
# #
# #     if(ParamIsDefault(q))
# #         q=0
# #     Else
# #         q=1
# #
# #
# #     #We want this function to be data folder aware
# #     #We'll do all our calculations in a specific data folder and then kill that folder at the end
# #     String savDF= GetDataFolder(1)    # Save current DF for restore.
# #     if( DataFolderExists("root:Cadzow_Data") )
# #         SetDataFolder root:Cadzow_Data
# #     else
# #         NewDataFolder/O/S root:Cadzow_Data    # Our stuff goes in here.
# #
# #
# #     #Timing
# #     Variable timerRef=startMSTimer
# #
# #     Variable N = len(signal);
# #     Variable L = floor(N*lfactor);
# #
# #     # T is the prediction matrix before filtering.
# #     Wave/C T = $Hankel(signal, N-L, L+1)
# #     T = conj(T)
# #
# #     if(M>(N-L))
# #         M = N-L
# #         print "M too large M set to: " + num2str(M)
# #
# #
# #     Variable i = 0
# #     Variable tol = 0
# #     Variable r = 0
# #
# #     print "Beginning Cadzow filtration, press ESC to abort, press CMD to check status."
# #
# #     for(i=0;i<iters;i+=1)
# #
# #         # decompose T
# #         #MatrixSVD Matrix
# #         MatrixSVD T
# #
# #         WAVE/C S = W_W
# #         WAVE/C U = M_U
# #         WAVE/C VT = M_VT
# #
# #         # check current rank
# #         tol = L*5e-16
# #         Duplicate/O S, S2
# #         S2 = (s>tol)
# #         r = sum(S2)
# #
# #         if(q || (GetKeyState(0) & 1))
# #             printf "Cadzow iteration %d (rank is %d, target is %d).\r", i, r,M
# #
# #
# #         if(r <= M)
# #             Printf "Successful completion: "
# #             break
# #         elif( r > M )
# #             #Filter the hankel matrix
# #             S = S*(p < M)
# #             Redimension/N=(-1,M) U
# #             Redimension/N=(M,-1) VT
# #             MatrixOp/C/O T = U x DiagRC(S,M,M) x VT
# #             # average to restore Hankel structure
# #             Wave recon_signal = $unHankelAvg(T)
# #             WAVE/C T = $Hankel(recon_signal,N-L,L+1)
# #
# #         if (GetKeyState(0) & 32)    # Is Escape key pressed now?
# #             Printf "User abort: "
# #             Break
# #
# #     EndFor
# #
# #     # need to extract data from matrix Tr
# #     T = conj(T)
# #
# #     #Move the results to the original data folder
# #     Duplicate/O $unHankelAvg(T), $(savDF+nameOfWave(signal)+"_cad")
# #     WAVE nSignal = $(savDF+nameOfWave(signal)+"_cad")
# #     SetDataFolder savDF    # Restore current DF.
# #
# #     #Clean up
# #     KillDataFolder root:Cadzow_Data
# #
# #     #Scale the new signal appropriately
# #     CopyScales/P signal, nSignal
# #
# #     #if the original signal was real, make the new signal real as well
# #     if((WaveType(signal) & 2^0) == 0)
# #         Redimension/R nSignal
# #
# #
# #     #Finish up the timing
# #     Variable microseconds = stopMSTimer(timerRef)
# #     Variable minutes = floor(microseconds/(60e6))
# #     Variable seconds = microseconds/(1e6)-minutes*60
# #
# #     if(!q)
# #         printf "Final rank is %d, target is %d, ", r,M
# #
# #
# #     Printf "%d iterations took ", i
# #     if(minutes > 1)
# #         Printf "%g minutes and ",minutes
# #     elif(minutes > 0)
# #         Printf "1 minute and "
# #
# #     Printf "%g seconds, for %g sec/iter.\r",seconds,microseconds/(1e6)/i
# #
# #     return  GetWavesDataFolder($(nameOfWave(signal)+"_cad"),2)
# # End
# #
# # STATIC Function/S unHankelAvg(Hankel)
# #     #A function that takes a Hankel matrix and returns the original signal
# #     #that it was formed from by averaging along the anti-diagonals
# #     Wave/C Hankel        #The matrix to be inverted
# #
# #     Variable numRows = DimSize(Hankel,0)
# #     Variable numCols = DimSize(Hankel,1)
# #
# #     #Make the signal to be returned, make sure to set to zero!
# #     Make/C/D/O/N=(numRows+numCols-1) mySignal=0
# #
# #     variable i=0,j=0
# #     Duplicate/C/O mySignal myNorm #Make the normalizing wave
# #     for(i=0;i<numRows;i+=1)
# #         for(j=0;j<numCols;j+=1)
# #             #Build up the signal and the norm
# #             mySignal[i+j]+=Hankel[i][j]
# #             myNorm[i+j] += complex(1,0)
# #         EndFor
# #     EndFor
# #     mySignal=mySignal/myNorm
# #     return  GetWavesDataFolder(mySignal,2)
# # End
# #
# #
# # Function OptimizeLPSVDCoefs(data,LPSVD_coefs,[ampcutoff,freqcutoff,dampcutoff,holdfreqphase])
# #     Wave data                                    #The original data
# #     Wave LPSVD_coefs                            #Parameters to optimize
# #     Variable ampcutoff, freqcutoff,dampcutoff    #Cutoff parameters to remove spurious values
# #     Variable holdfreqphase                        #hold the phases and frequencies constant during the fit
# #
# #     if(ParamIsDefault(ampcutoff))
# #         ampcutoff=0
# #
# #
# #     if(ParamIsDefault(freqcutoff))
# #         freqcutoff=0
# #
# #
# #     if(ParamIsDefault(dampcutoff))
# #         dampcutoff=0
# #
# #
# #     if(ParamIsDefault(holdfreqphase))
# #         holdfreqphase=0
# #
# #
# #     #Make a copy of the LPSVD_coefs, we'll use this wave later
# #     #to repack to optimized variables
# #     Duplicate/O LPSVD_coefs $("opt"+NameOfWave(LPSVD_coefs))
# #     WAVE newLPSVD_coefs = $("opt"+NameOfWave(LPSVD_coefs))
# #
# #     #Make a copy of data and remove the scaling from the copy.
# #     Duplicate/O data $("fit_"+nameofwave(data))
# #     WAVE newData = $("fit_"+nameofwave(data))
# #     SetScale/P x,0,1,"", newData
# #
# #     Variable numComponents = DimSize(LPSVD_coefs,0)
# #     variable i = 0
# #     String removedComponents = ""
# #     for(i=numComponents;i>0;i-=1)
# #         if((newLPSVD_coefs[i-1][%amps])^2<ampcutoff || (-LPSVD_coefs[i-1][%damps]/K_SPEED_OF_LIGHT/dimdelta(data,0)/np.pi) < dampcutoff || abs(newLPSVD_coefs[i-1][%freqs])<freqcutoff)
# #             removedComponents += num2istr(abs(newLPSVD_coefs[i-1][%freqs])/K_SPEED_OF_LIGHT/DimDelta(data,0)) +", "
# #             DeletePoints (i-1),1, newLPSVD_coefs
# #             numComponents-=1
# #
# #     EndFor
# #
# #     if(strlen(removedComponents))
# #         print "The following frequency components were removed: " + removedComponents
# #
# #
# #     #unpack LPSVD_coefs into a regular coefficient wave
# #     #Make use of the fact that only half of the coefficients are necessary
# #     #Also, set any frequency below some tolerance to zero and hold it there
# #     Variable numCoefs =  ceil(numComponents/2)
# #     Make/D/O/N=(numCoefs*4) myCoefs
# #     String HoldStr = ""
# #     for(i=0;i<numCoefs;i+=1)
# #         myCoefs[4*i] = 2*LPSVD_coefs[i][%amps]
# #         myCoefs[4*i+1] = LPSVD_coefs[i][%damps]
# #         if(abs(LPSVD_coefs[i][%freqs])<1e-14)
# #             myCoefs[4*i+2] = 0
# #             myCoefs[4*i+3] = 0
# #         Else
# #             myCoefs[4*i+2] = LPSVD_coefs[i][%freqs]
# #             myCoefs[4*i+3] = LPSVD_coefs[i][%phase]
# #
# #         if(holdfreqphase)
# #             HoldStr+="0011"
# #         Else
# #             HoldStr+="0000"
# #
# #     EndFor
# #
# #     #if there are an odd number of components the middle one is zero frequency
# #     if(numCoefs-floor(DimSize(LPSVD_coefs,0)/2))
# #         myCoefs[4*(numCoefs-1)] /= 2
# #
# #     Variable V_FitNumIters
# #     Variable V_FitMaxIters=200
# #     #do the optimization (we're using funcfit, so we're minimizing the chi^2)
# #     FuncFit/H=holdstr/ODR=2/N/W=2/Q decayingSinusoids, myCoefs, newData
# #
# #     print "Number of interations: "+num2str(V_FitNumIters)
# #     #Well use the newData wave to hold the fit, why not?
# #     newData = decayingSinusoids(myCoefs,p)
# #
# #     #return the scaling
# #     CopyScales/P data newData
# #
# #     #Repack
# #     for(i=0;i<numCoefs;i+=1)
# #         newLPSVD_coefs[i][%amps] = myCoefs[4*i]/2
# #         newLPSVD_coefs[i][%damps] = myCoefs[4*i+1]
# #         newLPSVD_coefs[i][%freqs] = myCoefs[4*i+2]
# #         newLPSVD_coefs[i][%phase] = myCoefs[4*i+3]
# #
# #         newLPSVD_coefs[2*numCoefs-i-1][%amps] = myCoefs[4*i]/2
# #         newLPSVD_coefs[2*numCoefs-i-1][%damps] = myCoefs[4*i+1]
# #         newLPSVD_coefs[2*numCoefs-i-1][%freqs] = -myCoefs[4*i+2]
# #         newLPSVD_coefs[2*numCoefs-i-1][%phase] = -myCoefs[4*i+3]
# #     EndFor
# # End
# #
# # Function decayingSinusoids(w,t)
# #     #w[i] = amp
# #     #w[i+1] = damp
# #     #w[i+2] = freq
# #     #w[i+3] = phase
# #     Wave w
# #     Variable t
# #
# #     Variable val=0
# #     Variable i=0
# #     Variable npts = len(w)
# #     for(i=0;i<npts;i+=4)
# #         val += w[i]*exp(t*w[i+1])*Cos(2*np.pi*w[i+2]*t+w[i+3])
# #     EndFor
# #
# #     return val
# # End
