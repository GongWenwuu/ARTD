function [Z, G, U, hist] = APG_RTD(X,Omega,Opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                       Alternating proximal gradient for ARTD                                    %
% \hat{\mathbf{x}}_{n}=\underset{\mathbf{x}_{n} \in \mathcal{X}_{n}}{\operatorname{argmin}}
% \left\langle\tilde{\mathbf{g}}, \mathbf{x}_{n}-\tilde{\mathbf{x}}_{n}\right\rangle
% +\frac{L_{\mathbf{x}_{n}}}{2}\left\|\mathbf{x}_{n}-\tilde{\mathbf{x}}_{n}\right\|_F^{2}+r_{n}(\mathbf{x}_{n})
%                                       This code was written by Wenwu Gong (2022.09)                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if isfield(Opts, 'epsilon'); epsilon = Opts.epsilon; else; epsilon = 1e-3; end 
if isfield(Opts,'Rpara'); Rpara = Opts.Rpara;  else; Rpara = 0.01; end
if isfield(Opts,'maxit'); maxit = Opts.maxit;  else; maxit = 500;  end
if isfield(Opts,'tol');   tol = Opts.tol;      else; tol = 1e-4;   end
if isfield(Opts,'gamma'); gamma = Opts.gamma;  else; gamma = 0;   end
if isfield(Opts,'phi');   phi = Opts.phi;      else; phi = 1;   end

N = ndims(X);
[Ginit,Uinit,~] = Initial(X.*Omega,Rpara,Opts.init);
L = cell(1,N); T = cell(1,N); alpha = zeros(1,N);
if isfield(Opts, 'flag')
    for n = 1:N
        Xmat = reshape(permute(X, [n 1:n-1 n+1:N]), size(X, n), []);
        if Opts.flag(n) == 1
            if strcmp(Opts.prior, 'stdc')
                L = constructL(size(X), {1, 2, 3}, 2, L);
            elseif strcmp(Opts.prior, 'toep')
                L{n} = constructT(size(X, n));
                L{n} = L{n}*L{n}';
            else
                L{n} = eye(size(X, n));
            end
            alpha(n) = norm(Xmat, 2)/(2*norm(L{n}, 2));
        elseif Opts.flag(n) == 0
            T{n} = constructT(size(Uinit{n},2));
            alpha(n) = norm(Xmat, 2)/(2*norm(T{n}*T{n}', 2));
        else
            alpha(n) = 0;
        end   
    end
else
    Opts.flag = 2.*ones(1,N);
end

obj0 = loss(X, Omega, Ginit, Uinit, L, T, alpha, Opts);

Usq = cell(1,N); 
for n = 1:N
    Usq{n} = Uinit{n}'*Uinit{n};
end

t0 = 1; niter = 0; Z = X.*Omega; 
G = Ginit; Gextra = Ginit; Lgnew = 1;
U = Uinit; Uextra = Uinit; LU0 = ones(N,1); LUnew = ones(N,1);
gradU = cell(N,1); wU = ones(N,1);
% figure('Position',get(0,'ScreenSize'));
% subplot(1,3,1);imshow(uint8(Z));title('incomplete tensor'); 

% time = tic;
for iter = 1:maxit
    % -- Core tensor updating --    
    gradG = gradientG(Gextra, U, Usq, Z); 
    Lg0 = Lgnew;
    Lgnew = lipG(Usq);
    if strcmp(Opts.init,'ntd')
        G = max(0, Gextra - gradG/Lgnew);
    elseif strcmp(Opts.init,'sntd')
        G = max(0,abs(Gextra - gradG/Lgnew) - Opts.beta/Lgnew);
    elseif strcmp(Opts.init,'rtd') || strcmp(Opts.init,'artd')   
        G = sign(Gextra - gradG/Lgnew).*max(0,abs(Gextra - gradG/Lgnew) - Opts.beta/Lgnew);   
    end    
    % -- Factor matrices updating --
    for n = 1:N
        gradU{n} = gradientU(Uextra, U, Usq, G, Z, L{n}, T{n}, alpha(n), n, Opts.flag(n)); 
        LU0(n) = LUnew(n);
        LUnew(n) = lipU(Usq, G, L{n}, T{n}, alpha(n), n, Opts.flag(n));
        if strcmp(Opts.init,'ntd') || strcmp(Opts.init,'rtd') || strcmp(Opts.init,'artd') 
            U{n} = max(0,Uextra{n} - gradU{n}/LUnew(n));
            %U{n} = Uextra{n} - gradU{n}/LUnew(n);
        elseif strcmp(Opts.init,'sntd')
            U{n} = max(0,Uextra{n} - gradU{n}/LUnew(n) - Opts.alpha(n)/LUnew(n));
        end    
        Usq{n} = U{n}'*U{n};  
    end
    Z_pre = Z;
    Z_new = ModalProduct_All(G,U,'decompress');
    Z = X.*Omega + gamma*(Z_pre - Z_new);
    Z(~Omega) = Z_new(~Omega);

    % -- diagnostics and reporting --
    objk = loss(X, Omega, G, U, L, T, alpha, Opts); 
    hist.obj(iter) = objk;
    relchange = norm(Z_new(:)-Z_pre(:))/norm(Z_pre(:));
    hist.err(1,iter) = relchange;
    relerr = TensorNorm(Omega.*(Z_new-X),'fro')/TensorNorm(Omega.*Z,'fro');
    hist.err(2,iter) = relerr;

    hist.rel(iter) = TensorNorm(X-Z_new,'fro')/TensorNorm(X,'fro');
    hist.rmse(iter) = sqrt((1/length(nonzeros(~Omega)))*norm(X(~Omega)-Z(~Omega),2)^2);
    hist.rse(iter) = norm(X(~Omega)-Z_new(~Omega),2)/norm(X(:),2);
    nmae = norm(X(~Omega)-Z(~Omega),1)/norm(X(~Omega),1);
    hist.nmae(iter) = nmae;
    
    % if mod(iter,10)==0 
    %      disp(['ARTD completed at ',int2str(iter),'-th iteration step within ',num2str(toc(time)),' seconds ']);
    %      fprintf('===================================\n');
    %      fprintf('Objective = %e\t, rel_DeltaX = %d\t,relerr = %d\t,NMAE = %d\n',objk, relchange,relerr,nmae);
    % end
    % subplot(1,3,2);plot(hist.rse);title('# iterations vs. RSEs');
    % subplot(1,3,3);imshow(uint8(Z));title('completed tensor');
    % axes('position',[0,0,1,1],'visible','off');
    % pause(0.1);

    % -- stopping checks and correction --
    if relerr < epsilon; niter = niter +1; else; niter = 0; end
    if relchange < tol || niter > 2
        break;
    end
    
    % -- extrapolation --      
    t = (1+sqrt(1+4*t0^2))/2;
    if objk >= obj0
        Gextra = Ginit;
        Uextra = Uinit;
    else
        w = (t0-1)/t;
        wG = min([w,0.999*sqrt(Lg0/Lgnew)]);
        Gextra = G + wG*(G - Ginit); 
        for n = 1:N
            wU(n) = min([w,0.9999*sqrt(LU0(n)/LUnew(n))]);
            Uextra{n} = U{n}+wU(n)*(U{n}-Uinit{n});
        end
        Ginit = G; Uinit = U; t0 = t; obj0 = objk; gamma = phi*gamma;
    end
    
end
Z(Omega) = X(Omega);

end