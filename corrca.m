function [W,ISC,Y,A,p]=corrca(X,varargin)
% [W,ISC,Y,A]=corrca(X) Correlated Component Analysis. Data volume X must
% be given as time-by-dimensions-by-subjects. The method finds projection
% vectors for the dimensions which maximize temporal correlation between
% subjects, denoted here as inter-subject correlation (ISC). We refere here
% to subjects for hitorical reasons. "subjects" could stand for any form of
% repeated versions of the time-by-dimension matrix.
%
% [W,ISC,Y,A]=corrca(X, ... 'version',vs) executes one of three version.
% vs=1: cohen et al. eNeuro 2016. vs=2: Parra 2017, which is faster in
% higher dimensions and more subjects. vs=3, JD: optimize SNR (variance of
% mean over mean of variance).
%
% [W,ISC,Y,A]=corrca(X, ... ,'shrinkage',gamma) uses shrinkage
% regularization with parameter gamma.
%
% [W,ISC,Y,A]=corrca(X, ... ,'tsvd',K) regularizes the within-subject
% covariance Rw by truncating its eigenvalue spectrum to the first K
% components.
%
% [W,ISC,Y,A,p]=corrca(X, ... ,'fixed',W) does not find the optimal W, but
% simply evaluates USC, Y and A for a given W, and computes p-values based
% on F-statistics. p-values are only valid if samples are uncorrelated in
% time and if W was not optimized for this data X. Use W=eye(D) to compute
% ISC and significance on original data. For correlated data the results
% are invalid, use shuffle statistics instead.

% Jul 16, 2017, Lucas Parra (c)
% Dec 18, 2017, Dmochowski, added tsvd regularization
% Dec 25, 2017, Parra, cleaned ISC computation for case of regularization 
% Jan 11, 2018, Parra, added p-value calculation. 
% Apr 30, 2018, Parra, handle rank-deficient data with TSVD 

% set detault options
version=2; % fastest version of the algorithms
gamma=0;   % no shrinkage
W=[];      % do compute W
K=size(X,2);       % don't truncate the eigs

for i=1:length(varargin)/2
    switch varargin{2*i-1}
        case 'version', version=varargin{2*i};
        case 'shrinkage', gamma=varargin{2*i};
        case 'fixed', W=varargin{2*i}; p=[]; % indicate that p is to be computed
        case 'tsvd', K=varargin{2*i}; gamma=0;  % don't combine shrinkage with tsvd
        otherwise, error(['Unknown option ' varargin{2*i-1} ])
    end
end

[T,D,N] = size(X);  % exemplars, dimensions, subjects

% compute within- and between-subject covariances
switch version
    
    case 1 % based on original definition of ISC (slow)
        Rkl = permute(reshape(cov(X(:,:)),[D N  D N]),[1 3 2 4]);
        Rw = sum(Rkl(:,:,1:N+1:N*N),3); % pooled over all subjects
        Rt = sum(Rkl(:,:,:),3);         % pooled over all pairs of subjects
        Rb = (Rt - Rw)/(N-1);
        
    case 2 % simplification that does not require all pairs (fast)
        Rw = 0; for l=1:N, Rw = Rw + cov(X(:,:,l)); end
        Rt = N^2*cov(mean(X,3));
        Rb = (Rt - Rw)/(N-1);
        
    case 3 % JD to maximize ensamble mean over total variation
        Rw = 0; for l=1:N, Rw = Rw + cov(X(:,:,l)); end
        Rt = N^2*cov(mean(X,3));
        Rb = Rt; % hack so I can reuse line with eig() below
        
end

% find projections W that maximize ISC
if isempty(W)
    K = min(rank(Rw),K); % handle rank deficient data. 
    if K<D 
        % with TSVD regularization
        [W,ISC]=eigs(regInv(Rw,K)*Rb,K);
    else       
        % with shrinkage regularization, or none if gamma=0           
        Rwreg=(1-gamma)*Rw+gamma*mean(diag(Rw))*eye(D);
        [W,ISC]=eig(Rb,Rwreg,'chol');
    end
    % make sure they are sorted by ISC and W normalized
    [~,indx]=sort(diag(real(ISC)),'descend'); W=W(:,indx); W=W*diag(1./sqrt(sum(W.^2)));
end

% compute ISC for fixed W, or recompute with unregularized Rw
ISC = diag(W'*Rb*W)./diag(W'*Rw*W);

% projections into corrca space
for l=N:-1:1, Y(:,:,l)=X(:,:,l)*W; end

% compute forward model; 
if K==D, A=Rw*W/(W'*Rw*W); % wont work for rank deficient data
else,    A=Rw*W*diag(1./diag(W'*Rw*W)); end
    
% Compute p-values based on F statistics, but only for fixed W.
if exist('p','var')
    
    % only valid for equal mean accross subjects (over-estimate of F otherwise)
    % S = (ISC+1/(N-1))./(1-ISC); 
    
    % using the exact formula for F-statistic (even for non-equal means)
    x = permute(Y,[2 1 3]);
    sw = sum(var(x,[],3),2)*(N-1);
    st = var(x(:,:),[],2)*(T*N-1);
    sb = st - sw;
    S = sb./sw;  
    
    df2=T*(N-1); df1=T-1;  F = df2/df1*S;
    p = fcdf(F,df1,df2,'upper');
end


end