function [W,ISC,Y,Ytest,ISCtest]=kcorrca(X,ktype,p,Xtest,model,Dy)
% [W,ISC,Y]=kcorrca(X) Kernel Correlated Component Analysis. Data volume X
% must be given as time-by-dimensions-by-subjects. The method finds
% projection vectors in the kernel space for the dimensions which maximize
% correlation between subjects, denoted here as inter-subject correlation
% (ISC). We refere here to subjects for hitorical reasons. "subjects" could
% stand for any form of repeated versions of the time-by-dimension matrix.
% Y is the data projected onto the new (non-linear kernel) component space.
%
% [W,ISC,Y]=kcorrca(X,kptype,p,Dy) same as above but using kernel type as
% specified in ktype. Can be one of 'Gaussian': K(x,y)=exp(-||x-y||^2/p);
% 'Polynomial':  K(x,y)=(x'*y)^p; 'tanh': K(x,y)=(x'*y).^p; In either case
% the third argument p is the recorresponding parameter. The default is
% 'Gaussian' with p=1. 
%
% [W,ISC,Y,Ytest,ISCtest]=kcorrca(X,kptype,p,Xtest) applies result to Xtest,
% which may have different number of subjects but must have same time and
% dimension.
%
% [W,ISC,Y,Ytest,ISCtest]=kcorrca(X,kptype,p,Xtest,model,Dy) allows to select
% the basis set that will be used for the kernel space. Options are
% model='full' which means that all time samples and subjects will be used
% as for the kernel space; or model='mean' which means that the time
% samples averaged across subjects are used. This model is more compact but
% does not capture variations across sujects. The default is model='mean'.
% Dy specifies the dimensionality of the subspace in which the kernel space
% is. Default is Dy = min([N Dx T]).
%

% (c) Lucas Parra, July 23, 2017
% Nov 27, more efficient computation of K

if nargin<2, ktype='Gaussian'; p=1; end

if nargin<5, model='mean'; end;

% determine number of subjects and dimensions of the kernel space
[T,Dx,N] = size(X); Dk=T; % Dk could be different if we change kernel()

% this is how many dimensions we we used when inverting Rw
if nargin<6, Dy= min([N Dx T]); end;
% not sure this is a good choice

% compute the kernel matrix
Xn = reshape(permute(X,[1,3,2]),T*N,Dx);
for l=N:-1:1
    tmp =  kernel(X(:,:,l),Xn,ktype,p);
    switch model
        case 'mean', K(:,:,l) = squeeze(mean(reshape(tmp,T,Dk,N),3));
        case 'full', K(:,:,l) = tmp;
        otherwise, error(['No model ' model ' implemented.'])
    end
end

% compute within- and between-subject covariances
Rw = 0; for l=1:N, Rw = Rw + cov(K(:,:,l)); end
Rt = N^2*cov(mean(K,3));
Rb = (Rt - Rw)/(N-1);

% find projections that maximize inter-subject correlation
% [W,ISC]=eig(Rb,Rw); % change code to assume rank=min(T,N)
% this is more robust
[W,ISC]=eigs(regInv(Rw,Dy)*Rb,Dy);

% sort by ISC
[ISC,indx]=sort(diag(ISC),'descend'); W=W(:,indx); W=W*diag(1./sqrt(sum(W.^2)));

% projections into kcorrca space
Y=[]; for l=N:-1:1, Y(:,:,l)=K(:,:,l)*W; end

% projections into kcorrca space
if nargout>3
    [Ttest,~,Ntest] = size(Xtest);
    for l=Ntest:-1:1
        tmp =  kernel(Xtest(:,:,l),Xn,ktype,p);
        switch model
            case 'mean', Ktest(:,:,l) = squeeze(mean(reshape(tmp,Ttest,Dk,N),3));
            case 'full', Ktest(:,:,l) = tmp;
        end
    end
    for l=Ntest:-1:1, Ytest(:,:,l)=Ktest(:,:,l)*W; end
end


if nargout>4
    Rw = 0; for l=1:Ntest, Rw = Rw + cov(Ktest(:,:,l)); end
    Rt = N^2*cov(mean(Ktest,3));
    Rb = (Rt - Rw)/(N-1);
    ISCtest = diag(W'*Rb*W)./diag(W'*Rw*W);
end

end

function K=kernel(X1,X2,ktype,p)

switch ktype
    case 'Gaussian'
        for i=size(X1,1):-1:1, 
            K(i,:) = exp(-sum((repmat(X1(i,:)',[1 size(X2,1)])-X2').^2)/p); end
    case 'Polynomial'
        K=abs(X1*X2').^p;
    case 'tanh'
        K=tanh(X1*X2'+ p);
    case 'linear'
        K=X1*X2';
    otherwise
        error([ktype ' - no such kernel type implemented.'])
end

end