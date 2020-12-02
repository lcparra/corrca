function [W,F,Y,Ytest]=klda(X,ktype,p,Xtest,Dy)
% [W,F,Y]=klda(X) Kernel Linear Discriminant Analysis. Data volume X must
% be given as class-by-dimensions-by-exemplars. The method finds projection
% vectors in the kernel space for the dimensions which maximize separation
% between classes.  Y is the data projected onto the new (non-linear
% kernel) component space.
%
% [W,F,Y]=klda(X,kptype,p,Dy) same as above but using kernel type as
% specified in ktype. Can be one of 'Gaussian': K(x,y)=exp(-||x-y||^2/p);
% 'Polynomial':  K(x,y)=(x'*y)^p; 'tanh': K(x,y)=(x'*y).^p; In either case
% the third argument p is the recorresponding parameter. The default is
% 'Gaussian' with p=1. 

% [W,F,Y,Ytest]=klda(X,kptype,p,Xtest) applies result to Xtest, which may
% have different number of exemplars but must have same number of classes
% and dimension.
%
% [W,F,Y,Ytest]=klda(X,kptype,p,Xtest,Dy) allows to select the
% basis set that will be used for the kernel space. Options are
% model='full' which means that all classes and exemplars will be used as
% for the kernel space; or model='mean' which means that the class means
% are used. This model is more compact but does not capture variations
% within a class. The default is model='full'. Dy specifies the
% dimensionality of the subspace in which the kernel space is.
%

% (c) Lucas Parra, July 27, 2017

if nargin<2, ktype='Gaussian'; p=1; end

% determine number of subjects and dimensions of the kernel space
[C,Dx,N] = size(X); Dk=C; % Dk could be different if we change kernel()

% this is how many dimensions we we used when inverting Sw
% if nargin<4, Dy= min([N Dx C]); end;
% commented out because I dont think these are a good choice
    
% compute the kernel matrix
Xn = reshape(permute(X,[1,3,2]),C*N,Dx);
K = zeros(C,Dk*N,N);
for l=1:N, K(:,:,l) =  kernel(X(:,:,l),Xn,ktype,p); end
    
if 1 % use the code that I have from kCCA (work great!!!)
    
    % compute within- and between class covariances
    Rw = 0; for l=1:N, Rw = Rw + cov(K(:,:,l)); end
    Rt = N^2*cov(mean(K,3));
    Rb = (Rt - Rw)/(N-1);
    % find projections that maximize between over within scatter
    [W,F]=eigs(regInv(Rw,Dy)*Rb,Dy);

else % use the code that I have from LDA
    
    % class and total mean
    kcm=mean(K,3); ktm = repmat(mean(kcm),C,1);
    
    % compute Scatter matrices (within, between, total)
    Sw = 0; for l=1:N, tmp=K(:,:,l)-kcm; Sw = Sw + tmp'*tmp/C/N; end;
    St = 0; for l=1:N, tmp=K(:,:,l)-ktm; St = St + tmp'*tmp/C/N; end;
    Sb = St - Sw;
    
    % find projections that maximize between over within scatter
    [W,F]=eigs(regInv(Sw,Dy)*Sb,Dy);
    
end
% sort by ISC
[F,indx]=sort(diag(F),'descend'); W=W(:,indx); W=W*diag(1./sqrt(sum(W.^2)));

% projections into kCCA space
Y=[]; for l=N:-1:1, Y(:,:,l)=K(:,:,l)*W; end

% projections into kCCA space
if nargin>3
    for l=size(Xtest,3):-1:1
        Ytest(:,:,l) =  kernel(Xtest(:,:,l),Xn,ktype,p)*W;
    end
end

end

function K=kernel(X1,X2,ktype,p)

switch ktype
    case 'Gaussian'
        for i=size(X1,1):-1:1, 
            K(i,:) = exp(-sum((repmat(X1(i,:)',[1 size(X2,1)])-X2').^2)/p); end
    case 'Polynomial'
        K=(X1*X2').^p;
    case 'tanh'
        K=tanh(X1*X2'+ p);
    otherwise
        error([ktype ' - no such kernel type implemented.'])
end

end