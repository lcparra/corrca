function [W,ISC,Y,A,p]=pca_mean(X,varargin)
% this code is a hacked version of corrca.m to implement the suggestion
% to do PCA on the mean across subjects as a benchmark to corrca. 
% this code can be called like the original corrca.m, but probably wont
% work for most parameters (version, regularization, rank deficient data).

% May 28, 2018, Lucas Parra


% set detault options
version=2; % fastest version of the algorithms
gamma=0;   % no shrinkage
W=[];      % do compute W
K=size(X,2); % don't truncate the eigs

for i=1:length(varargin)/2
    switch varargin{2*i-1}
        case 'version', version=varargin{2*i};
        case 'shrinkage', gamma=varargin{2*i};
        case 'fixed', W=varargin{2*i}; p=[]; % indicate that p is to be computed
        case 'tsvd', K=varargin{2*i}; gamma=0;  % don't combine shrinkage with tsvd
        otherwise error(['Unknown option ' varargin{2*i-1} ])
    end
end

[T,D,N] = size(X);  % exemplars, dimensions, subjects

% compute within- and between-subject covariances
Rw = 0; for l=1:N, Rw = Rw + cov(X(:,:,l)); end
Rt = N^2*cov(mean(X,3));
Rb = (Rt - Rw)/(N-1);

% find projections W that maximize ISC
if isempty(W)
    
    % [W,ISC]=eig(Rb,Rwreg,'chol');    % ***    here is the hack    ***
    [W,~]=eig(Rt);                     % *** PCA on the mean instad ***

    % compute ISC 
    ISC = diag(W'*Rb*W)./diag(W'*Rw*W);
    
    % make sure they are sorted by ISC and W normalized
    [ISC,indx]=sort(real(ISC),'descend'); W=W(:,indx); W=W*diag(1./sqrt(sum(W.^2)));

end

% compute ISC for fixed W, or recompute with unregularized Rw
ISC = diag(W'*Rb*W)./diag(W'*Rw*W);

% projections into corrca space
for l=N:-1:1, Y(:,:,l)=X(:,:,l)*W; end

% compute forward model
A=Rw*W*inv(W'*Rw*W);

end