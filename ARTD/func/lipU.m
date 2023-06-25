function L = lipU(U, G, Lap, Tem, alpha, mode, flag)
% L_{\mathbf{U}_{n}} =
% \left\|{\mathbf{G}_{(n)}} {\mathbf{V}}^{\mathrm{T}}_{n} {\mathbf{V}}\mathbf{G}_{(n)}^{\mathrm{T}}\right\|_{\mathrm{F}} 
% \left\| \alpha \mathbf{L} \right\|_{\mathrm{F}}
% \left\| \alpha \mathbf{T}\mathbf{T}^{\mathrm{T}} \right\|_{\mathrm{F}}

Bsq = Matrixization(G,U,mode,'decompress')*Unfold(G,mode)';

if flag == 1
    L = norm(Bsq,2) + alpha*norm(Lap,2);
elseif flag == 0
    L = norm(Bsq,2) + alpha*norm((Tem'*Tem),2);
else
    L = norm(Bsq,2);
end

end

function X_unf = Unfold(X,mode)
    N = ndims(X);
    X_unf = reshape(permute(X, [mode 1:mode-1 mode+1:N]),size(X,mode),[]);
end