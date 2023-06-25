function obj = loss(X, Omega, G, USet, L, T, alpha, Opts)

    Z = ModalProduct_All(G,USet,'decompress');    
    obj = 0.5*TensorNorm(Omega.*(X-Z),'fro')^2; 
    if Opts.beta > 0
        obj = obj + Opts.beta*TensorNorm(G,1);
    end
    for n = 1:ndims(X)
        if alpha(n) > 0
            if Opts.flag(n) == 1
                obj = obj + 0.5*alpha(n)*trace(USet{n}'*L{n}*USet{n});  
            elseif Opts.flag(n) == 0
                obj = obj + 0.5*alpha(n)*norm(T{n}*USet{n},'fro')^2;
            end
        end    
    end
    
end