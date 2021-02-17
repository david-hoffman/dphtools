function  [X,info,perf] = SMarquardt(fun,par, x0, opts, B0)
%SMarquardt  Secant version of Marquardt's method for least squares.
%  Find  xm = argmin{F(x)} , where  x = [x_1, ..., x_n]  and
%  F(x) = .5 * sum(f_i(x)^2) .
%  The functions  f_i(x) (i=1,...,m)  must be given by a MATLAB
%  function with declaration
%            function  f = fun(x, par)
%  par  can e.g. be an array with coordinates of data points,
%  or it may be dummy.
%
%  Call
%      [X, info {,perf}] = SMarquardt(fun,par, x0, opts {,B0}) 
%
%  Input parameters
%  fun  :  String with the name of the function.
%  par  :  Parameters of the function.  May be empty.
%  x0   :  Starting guess for  x .
%  opts :  Vector with five elements:
%          opts(1) used in starting guess for Marquardt parameter: 
%              mu = opts(1) * max(A0(i,i))  with  A0 = B'*B , 
%          where  B  is the initial approximation to the Jacobian.
%          opts(2:4) used in stopping criteria
%              ||B'*f||inf <= opts(2)                   or 
%              ||dx||2 <= opts(3)*(opts(3) + ||x||2)    or
%              no. of iteration steps exceeds  opts(4) . 
%          opts(5) = Step used in difference approximation to the 
%              Jacobian.  Not used if  B0  is present.
%  B0   :  (optional).  If given, then initial approximation to J.
%          If  B0 = [], then it is replaced by  eye(m,n) .
%
%  Output parameters
%  X    :  If  perf  is present, then array, holding the iterates
%          columnwise.  Otherwise, computed solution vector.
%  info :  Performance information, vector with 8 elements:
%          info(1:4) = final values of 
%              [F(x)  ||F'||inf  ||dx||2  mu/max(A(i,i))] ,
%            where  A = B'* B .
%          info(5) = no. of iteration steps
%          info(6) = 1 :  Stopped by small gradient
%                    2 :  Stopped by small x-step
%                    3 :  Stopped by  kmax
%                    4 :  Problems, indicated by printout
%          info(7) = no. of function evaluations
%          info(8) = no. of difference approximations to the Jacobian  
%  perf :  (optional). If present, then array, holding 
%            perf(1,:) = values of  F(x)
%            perf(2,:) = values of  || F'(x) ||inf
%            perf(3,:) = mu-values.

%  Hans Bruun Nielsen,  IMM, DTU.  99.06.10 / 00.09.04

   %  Check function call
   nin = nargin;
   [xb m n fb] = check(fun,par,x0,opts,nin);   fcl = 1;
   %  Initialize
   if  nin > 4
     reappr = 0;  sB = size(B0);
     if      sum(sB) == 0,  B = eye(m,n);
     elseif  any(sB ~= [m n])
       error('Dimension of B0 do not match  f  and  x')
     else,   B = B0; end
   else
     B = Dapprox(fun,par,xb,fb,opts(5));
     reappr = 1;   fcl = fcl + n;
   end  
   mu = opts(1) * max(sum(B .* B));
   Fb = (fb'*fb)/2;    kmax = opts(4);
   Trace = nargout > 2;
   if  Trace,  X = zeros(n,kmax+1);  perf = zeros(3,kmax+1); end
   k = 0;   nu = 2;   stop = 0;
   K = max(10,n);   updB = 0;   updx = 1;  % updx changed 00.09.04

   while  ~stop
     if  reappr & ((updx & nu > 16) | (updB == K))
       % Recompute difference approximation
       B = Dapprox(fun,par,xb,fb,opts(5));
       reappr = reappr + 1;   fcl = fcl + n;
       nu = 2;   updB = 0;   updx = 0; 
     end
     g = B'*fb;    ng = norm(g,inf);   k = k + 1;
     if  Trace,  X(:,k) = xb;   perf(:,k) = [Fb ng mu]'; end
     if  ng <= opts(2),  stop = 1;  
     else    %  Compute Marquardt step
       h = (B'*B + mu*eye(n))\(-g);
       nh = norm(h);   nx = opts(3) + norm(xb);
       if      nh <= opts(3)*nx,  stop = 2;
       elseif  nh >= nx/eps
         stop = 4;   disp('Marquardt matrix is (almost) singular')
       end
     end
     if  ~stop
       xnew = xb + h;    h = xnew - xb; 
       fn = feval(fun, xnew,par);   fcl = fcl + 1;
       Fn = (fn'*fn)/2;
       if  updx | (Fn < Fb)    % Update  B     
         B = B + ((fn - fb - B*h)/(h'*h)) * h';
         updB = updB + 1;
       end
       %  Update  x  and  mu
       if  Fn < Fb
         dL = .5*(h'*(mu*h - g));   rho = max(0, (Fb - Fn)/dL);
         mu = mu * max(1/3, (1 - (2*rho - 1)^7));   nu = 2; 
         xb = xnew;   Fb = Fn;   fb = fn;   updx = 1;
       else
         mu = mu*nu;  nu = 2*nu;
       end 
       if  k > kmax,  stop = 3; end 
     end  
   end
   %  Set return values
   if  Trace
     X = X(:,1:k);   perf = perf(:,1:k);
   else,  X = xb;  end
   info = [Fb  ng  nh  mu/max(sum(B .* B))  k-1  stop  fcl  reappr];

% ==========  auxiliary functions  =================================

function  [x,m,n, f] = check(fun,par,x0,opts,nin)
%  Check function call
   sx = size(x0);   n = max(sx);
   if  (min(sx) > 1)
       error('x0  should be a vector'), end
   x = x0(:);   f = feval(fun,x,par);
   sf = size(f);
   if  sf(2) ~= 1
       error('f  must be a column vector'), end
   m = sf(1);
%  Thresholds
   if  nin > 4,  nopts = 4;  else,  nopts = 5; end
   if  length(opts) < nopts
       tx = sprintf('opts  must have %g elements',nopts);
       error(tx), end
   if  length(find(opts(1:nopts) <= 0))
       error('The elements in  opts  must be strictly positive'), end

function  B = Dapprox(fun,par,x,f,delta)
%  Difference approximation to Jacobian
   n = length(x);   B = zeros(length(f),n);
   for  j = 1 : n
     z = x;  z(j) = x(j) + delta;   d = z(j) - x(j);
     B(:,j) = (feval(fun,z,par) - f)/d;
   end 