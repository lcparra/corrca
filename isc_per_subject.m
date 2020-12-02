function ISC=isc_per_subject(Y)
% ISC=isc_per_subject(Y) computes inter-subject correlation between each 
% subject and all others, for each column in Y, which is a volume with 
% T exemplars, D dimensions, and N subjects. ISC has dimensions 
%
% see corrca()

% Sep 11, 2018, Jens Madsen & Parra, memory efficient implementation for large N
% Oct 24, 2018, Parra, standalone function combining new and old implementation. 

[~,D,N] = size(Y);  % exemplars, dimensions, subjects

if N>30    % faster and less memory intensive for large N

    for i = D:-1:1
        R(:,:,i) = cov(squeeze(Y(:,i,:)));
        r(:,i) = diag(R(:,:,i));
    end
    
    for i=N:-1:1
        j = setdiff(1:N,i); % all subjects but the i-th
        Rij = squeeze(R(i,j,:));
        Rb = sum(Rij    + Rij   );  % betweeen subject
        Rw = sum(r(j,:) + r(i,:));  % within subject
        ISC(i,:) = Rb./Rw;
    end
    
else     % perhaps more transparent implementation and faster for small N

    Rij = permute(reshape(cov(Y(:,:)),[D N  D N]),[1 3 2 4]);
    
    % Compute ISC resolved by subject, see Cohen et al.
     for i=N:-1:1
        Rw=0; for j=1:N, if i~=j, Rw = Rw+1/(N-1)*(Rij(:,:,i,i)+Rij(:,:,j,j)); end; end
        Rb=0; for j=1:N, if i~=j, Rb = Rb+1/(N-1)*(Rij(:,:,i,j)+Rij(:,:,j,i)); end; end
        ISC(i,:) = diag(Rb)./diag(Rw);
    end
    
end
