function [W,F]=cca(X,v);
% [W,F]=lda(X) Linear Discriminant Analysis. The method finds projection
% vectors for the dimensions which maximize the variance between the class
% means over the mean of the within class variance, i.e. the F statistics,
% which should be more approritately call the R statistics as LDA for 2
% classes introduced by Fisher was generalized to muliple classes by Rao.
% Data volume X must be given as classes x dimensions x exemplars.

% (c) Lucas Parra, July 16, 2017

if nargin<2, v=2; end % by dedault do the faster version

[C,D,N] = size(X);  % classes, dimensions, exemplars

% class and total mean
xcm=mean(X,3); xtm = repmat(mean(xcm),C,1); 

% compute Scatter matrices (within, between, total)
Sw = 0; for l=1:N, tmp=X(:,:,l)-xcm; Sw = Sw + tmp'*tmp/C/N; end;
St = 0; for l=1:N, tmp=X(:,:,l)-xtm; St = St + tmp'*tmp/C/N; end;
Sb = St - Sw;

% find projections that maximize between over withith class scatter. 
[W,F]=eig(Sb,Sw);

% sort based on F statistics
[F,indx]=sort(diag(F),'descend'); W=W(:,indx); W=W*diag(1./sqrt(sum(W.^2)));

end