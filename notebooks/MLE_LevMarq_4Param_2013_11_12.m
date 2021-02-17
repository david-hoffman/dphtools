function [pn,CRLB,u,rChi,delta,convergeflag,pval,chi2MLE]=MLE_LevMarq_4Param_2013_11_12(PSF1,X,Y,PSFWidth,p_initial,maxIter)
%A function to perform maximum likelihood fitting of single emitter spots.  This function fits the integrated intensity, the x-centroid, y-centroid, and
%per pixel background under a symmetric 2D Gaussian PSF model. This function uses the levenberg-marqquet algorithm to minimize the Chi-square MLE parameter


% Input Parameters
% PSF1 = a N X N X 1 array of the photon-converted image data to fit
% X and Y = are the output from [Y,X]=meshgrid([1:1:2*WindowHalfWidth+1],[1:1:2*WindowHalfWidth+1]);
%   where WindowHalfWidth is defined as in findPeaks_2013_11_25.
% PSFWidth = the ideal sigma of the gaussian used for fitting
% p_initial = a 1 X 4 vector containing the initial guesses for the fit parameters
% maxIter = the maximum number of iterations for the Newton solver to take


%Output Parameters
% pn = a 1 X 4 vector of the fitted parameters
%     pn(1) = the integrated intensity of the spot
%     pn(2) = the x-centroid of the spot
%     pn(3) = the y-centroid of the spot
%     pn(4) = the per pixel background of the spot
% CRLB = the Cramer Rao Lower Bound of the parameter fits - as defined in Smith, CS et. al. Nature Methods 2010
% u = the fit image
% rChi = the sum of the differences between the fit image u and the raw image data
% delta = an iters X 4 matrix displaying the amount that each variable is incrementated by in each iteration of the Newton solver
% convergeflag = a flag to indicate algorithm convergence. If change in the chi^2 value is less than funTol then convergeflag = 1, else convergeflag = 0
% pval = supposed to be a probably test for goodness of fit, but is not yet implemented correctly
% chi2MLE = the chi^2 value at the last iteration

Io=p_initial(1); % initial guess for the integrated intensity of the spot
xo=p_initial(2); % initial guess for the x-centroid of the spot
yo=p_initial(3); % initial guess for the y-centroid of the spot
bg=p_initial(4); % initial guess for the per pixel background of the spot
sigx=PSFWidth; % user supplied value for the x-sigma
sigy=PSFWidth; % user supplied value for the y-sigma

%Initialize variables
rChi=0;
delta=zeros(4,maxIter);
diagnostics=0; %Turn this on to plot the fit data
funTol = 1e-5; %The minimum threshold to define how much the Chi-square value needs to decrease in order to take a step
convergeflag=0;

%Define image model
Ex=(1./2).*(-erf((-0.5 + X - xo)./(sqrt(2).*sigx)) + erf((0.5 + X - xo)./(sqrt(2).*sigx)));
Ey=(1./2).*(-erf((-0.5 + Y - yo)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo)./(sqrt(2).*sigy)));
u=sqrt(bg^2)+sqrt(Io^2)*Ex.*Ey;
chi2MLE=2*sum(u(:)-PSF1(:))-2.*sum(PSF1(:).*log(u(:)./PSF1(:)));

for i=1:maxIter
    %Calculate 1st derivatives
    sqrtIosq=sqrt(Io^2);
    du1_dIo=Io/sqrtIosq.*Ex.*Ey;
    du1_dxo=(1./2).*sqrtIosq.*(sqrt(2./pi)./(exp((-0.5 + X - xo).^2./(2.*sigx.^2)).*sigx) - sqrt(2./pi)./(exp((0.5 + X - xo).^2./(2.*sigx.^2)).*sigx)).*Ey;
    du1_dyo=(1./2).*sqrtIosq.*(sqrt(2./pi)./(exp((-0.5 + Y - yo).^2./(2.*sigy.^2)).*sigy) - sqrt(2./pi)./(exp((0.5 + Y - yo).^2./(2.*sigy.^2)).*sigy)).*Ex;
    du1_dbg=bg/sqrt(bg^2);
    
    %Calculate Jacobian
    PSFdivu=PSF1(:)./u(:);
    B_k(1,1)=-sum((1-PSFdivu).*du1_dIo(:));
    B_k(2,1)=-sum((1-PSFdivu).*du1_dxo(:));
    B_k(3,1)=-sum((1-PSFdivu).*du1_dyo(:));
    B_k(4,1)=-sum((1-PSFdivu).*du1_dbg(:));
    
    %Calculate Hessian - setting second derivatives to zeros
    lbda=0;
    PSFdivusq=PSF1(:)./u(:).^2;
    
    A_kl(1,1)=sum(du1_dIo(:).*du1_dIo(:).*PSFdivusq)*(1+lbda);
    A_kl(2,1)=sum(du1_dIo(:).*du1_dxo(:).*PSFdivusq);
    A_kl(3,1)=sum(du1_dIo(:).*du1_dyo(:).*PSFdivusq);
    A_kl(4,1)=sum(du1_dIo(:).*du1_dbg(:).*PSFdivusq);
    
    A_kl(1,2)=A_kl(2,1);
    A_kl(2,2)=sum(du1_dxo(:).*du1_dxo(:).*PSFdivusq)*(1+lbda);
    A_kl(3,2)=sum(du1_dxo(:).*du1_dyo(:).*PSFdivusq);
    A_kl(4,2)=sum(du1_dxo(:).*du1_dbg(:).*PSFdivusq);
    
    A_kl(1,3)=A_kl(3,1);
    A_kl(2,3)=A_kl(3,2);
    A_kl(3,3)=sum(du1_dyo(:).*du1_dyo(:).*PSFdivusq)*(1+lbda);
    A_kl(4,3)=sum(du1_dyo(:).*du1_dbg(:).*PSFdivusq);
    
    A_kl(1,4)=A_kl(4,1);
    A_kl(2,4)=A_kl(4,2);
    A_kl(3,4)=A_kl(4,3);
    A_kl(4,4)=sum(du1_dbg(:).*du1_dbg(:).*PSFdivusq)*(1+lbda);
    
    delta(:,i)=pinv(A_kl)*B_k;
    
    Io_test=Io+delta(1,i);
    xo_test=xo+delta(2,i);
    yo_test=yo+delta(3,i);
    bg_test=bg+delta(4,i);
    
    %     Test whether update gives a more favorable solution as measured by a greater likelihood
    passed=0;
    t=1;
    while (~passed && t<=20 && ~convergeflag) %Try to refine the step at most 20 times before quitting
        %Recalculate the objective function at the proposed parameter location
        Ex_test=(1./2).*(-erf((-0.5 + X - xo_test)./(sqrt(2).*sigx)) + erf((0.5 + X - xo_test)./(sqrt(2).*sigx)));
        Ey_test=(1./2).*(-erf((-0.5 + Y - yo_test)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo_test)./(sqrt(2).*sigy)));
        u_test=sqrt(bg_test^2)+sqrt(Io_test^2).*Ex_test.*Ey_test;
        chi2MLE_test=2*sum(u_test(:)-PSF1(:))-2.*sum(PSF1(:).*log(u_test(:)./PSF1(:)));
        passed=chi2MLE_test<=(chi2MLE+funTol);%Compare with the previous value
        %If the value doesn't decrease (or doesn't stay within funTol of the previous value - suggesting convergence), try a smaller step
        if (~passed && t<20)
            lbda=2^t;
            t=t+1;
            A_kl(1,1)=sum(du1_dIo(:).*du1_dIo(:).*PSFdivusq)*(1+lbda);
            A_kl(2,2)=sum(du1_dxo(:).*du1_dxo(:).*PSFdivusq)*(1+lbda);
            A_kl(3,3)=sum(du1_dyo(:).*du1_dyo(:).*PSFdivusq)*(1+lbda);
            A_kl(4,4)=sum(du1_dbg(:).*du1_dbg(:).*PSFdivusq)*(1+lbda);
            delta(:,i)=pinv(A_kl)*B_k;
            Io_test=Io+delta(1,i);
            xo_test=xo+delta(2,i);
            yo_test=yo+delta(3,i);
            bg_test=bg+delta(4,i);
        else if t>=20
                warning('The smallest step size has been reached, but algorithm has not converged')
                pn=zeros(4,1);
                CRLB=zeros(4,1);
                u=PSF1*0;
                rChi=0;
                delta=0;
                convergeflag=0;
                pval=-1;
                chi2MLE=-1;                
                return
            else %Update parameter values
                Io=Io_test;
                xo=xo_test;
                yo=yo_test;
                bg=bg_test;
                Ex=Ex_test;
                Ey=Ey_test;
                u=u_test;
                convergeflag=((chi2MLE-chi2MLE_test)/chi2MLE)<funTol;
            end
        end
    end
    chi2MLE=chi2MLE_test;
end
pn=[sqrt(Io^2),xo,yo,sqrt(bg^2)];

%Calculate Fisher information matrix
FSHR(1,1)=sum(du1_dIo(:).*du1_dIo(:)./u(:));
FSHR(1,2)=sum(du1_dIo(:).*du1_dxo(:)./u(:));
FSHR(1,3)=sum(du1_dIo(:).*du1_dyo(:)./u(:));
FSHR(1,4)=sum(du1_dIo(:).*du1_dbg(:)./u(:));

FSHR(2,1)=FSHR(1,2);
FSHR(2,2)=sum(du1_dxo(:).*du1_dxo(:)./u(:));
FSHR(2,3)=sum(du1_dxo(:).*du1_dyo(:)./u(:));
FSHR(2,4)=sum(du1_dxo(:).*du1_dbg(:)./u(:));

FSHR(3,1)=FSHR(1,3);
FSHR(3,2)=FSHR(2,3);
FSHR(3,3)=sum(du1_dyo(:).*du1_dyo(:)./u(:));
FSHR(3,4)=sum(du1_dyo(:).*du1_dbg(:)./u(:));

FSHR(4,1)=FSHR(1,4);
FSHR(4,2)=FSHR(2,4);
FSHR(4,3)=FSHR(3,4);
FSHR(4,4)=sum(du1_dbg(:).*du1_dbg(:)./u(:));

%Calculate CRLB
FSHRinv=inv(FSHR);
CRLB=diag(FSHRinv);

rChi=sum((PSF1(:)-u(:)).^2./PSF1(:));
pval = 0;

%diagnostics provides graphical plots of the function behavior in the neighborhood of the final parameter values
if diagnostics==1
    %Check molecule fit compared to raw data
    Ex=(1./2).*(-erf((-0.5 + X - xo)./(sqrt(2).*sigx)) + erf((0.5 + X - xo)./(sqrt(2).*sigx)));
    Ey=(1./2).*(-erf((-0.5 + Y - yo)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo)./(sqrt(2).*sigy)));
    u=sqrt(bg^2)+sqrt(Io^2).*Ex.*Ey;
    subplot(1,2,1)
    hold on
    imagesc(u)
    scatter(pn(3),pn(2),'bo','filled')
    axis equal
    title('fit image')
    subplot(1,2,2)
    hold on
    imagesc(PSF1)
    scatter(pn(3),pn(2),'bo','filled')
    axis equal
    title('original image')
    Irange=[sqrt(Io^2)-1000:1:sqrt(Io^2)+1000]';
    figure
    subplot(2,2,1)
    hold on
    for ii=1:length(Irange)
        Ex=(1./2).*(-erf((-0.5 + X - xo)./(sqrt(2).*sigx)) + erf((0.5 + X - xo)./(sqrt(2).*sigx)));
        Ey=(1./2).*(-erf((-0.5 + Y - yo)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo)./(sqrt(2).*sigy)));
        u=sqrt(bg^2)+sqrt(Irange(ii)^2).*Ex.*Ey;
        chi2MLE_I(ii)=2*sum(u(:)-PSF1(:))-2.*sum(PSF1(:).*log(u(:)./PSF1(:)));
    end
    plot(Irange,chi2MLE_I,'b')
    plot(sqrt(Io^2),chi2MLE,'bo')
    xlabel('Io')
    ylabel('Log-Likelihood')
    subplot(2,2,2)
    hold on
    xrange=[xo-2:.01:xo+2]';
    for ii=1:length(xrange)
        Ex=(1./2).*(-erf((-0.5 + X - xrange(ii))./(sqrt(2).*sigx)) + erf((0.5 + X - xrange(ii))./(sqrt(2).*sigx)));
        Ey=(1./2).*(-erf((-0.5 + Y - yo)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo)./(sqrt(2).*sigy)));
        u=sqrt(bg^2)+sqrt(Io^2).*Ex.*Ey;
        chi2MLE_x(ii)=2*sum(u(:)-PSF1(:))-2.*sum(PSF1(:).*log(u(:)./PSF1(:)));
    end
    plot(xrange,chi2MLE_x,'r')
    plot(xo,chi2MLE,'ro')
    xlabel('xo')
    ylabel('Log-Likelihood')
    subplot(2,2,3)
    hold on
    yrange=[yo-2:.01:yo+2]';
    for ii=1:length(yrange)
        Ex=(1./2).*(-erf((-0.5 + X - xo)./(sqrt(2).*sigx)) + erf((0.5 + X - xo)./(sqrt(2).*sigx)));
        Ey=(1./2).*(-erf((-0.5 + Y - yrange(ii))./(sqrt(2).*sigy)) + erf((0.5 + Y - yrange(ii))./(sqrt(2).*sigy)));
        u=sqrt(bg^2)+sqrt(Io^2).*Ex.*Ey;
        chi2MLE_y(ii)=2*sum(u(:)-PSF1(:))-2.*sum(PSF1(:).*log(u(:)./PSF1(:)));
    end
    plot(yrange,chi2MLE_y,'g')
    plot(yo,chi2MLE,'go')
    xlabel('yo')
    ylabel('Log-Likelihood')
    subplot(2,2,4)
    hold on
    bgrange=[sqrt(bg^2)-10:.01:sqrt(bg^2)+10]';
    for ii=1:length(bgrange)
        Ex=(1./2).*(-erf((-0.5 + X - xo)./(sqrt(2).*sigx)) + erf((0.5 + X - xo)./(sqrt(2).*sigx)));
        Ey=(1./2).*(-erf((-0.5 + Y - yo)./(sqrt(2).*sigy)) + erf((0.5 + Y - yo)./(sqrt(2).*sigy)));
        u=sqrt(bgrange(ii)^2)+sqrt(Io^2).*Ex.*Ey;
        chi2MLE_bg(ii)=2*sum(u(:)-PSF1(:))-2.*sum(PSF1(:).*log(u(:)./PSF1(:)));
    end
    plot(bgrange,chi2MLE_bg,'c')
    plot(sqrt(bg^2),chi2MLE,'co')
    xlabel('bg')
    ylabel('Log-Likelihood')
end

