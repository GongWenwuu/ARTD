function L = lipG(U)
% \| \mathbf{U}_{1}^{\mathrm{T}} {\mathbf{U}_{1}}\|_{\mathrm{F}} \| \mathbf{V}_{1}^{\mathrm{T}} {\mathbf{V}_{1}}\|_{\mathrm{F}}
    L = 1;
    for n = 1:length(U)
        L = L*norm(U{n},2);
    end
end