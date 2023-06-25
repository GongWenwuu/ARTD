function [G0,U0,Rank] = Initial(X,para,init)
    
    load('s.mat','s');
    rng(s);
    N = ndims(X);
    Nway = size(X);
    U0 = cell(1,N);
    if strcmp(init,'ntd') || strcmp(init,'sntd')
        if length(para) == 1 
            [U0,~] = myRank(X,para,'tol');
        else
            for n = 1:N
                U0{n} = max(0,orth(randn(Nway(n),para(n))));
            end
        end
        G0 = ModalProduct_All(X, U0, 'compress');
        Rank = size(G0);
    elseif strcmp(init,'rtd')
        [G0, U0, Rank] = HOSVD(X,1e-3,para);
        Xnorm = 0.5*TensorNorm(X,'fro');
        G0 = (G0/TensorNorm(G0,'fro')*Xnorm^(1/(N+1)));
    elseif strcmp(init,'hosvd')
         [U0,Rank] = myRank(X,para,'truncated');
         G0 = ModalProduct_All(X, U0, 'compress');
    elseif strcmp(init,'artd')
        N = ndims(X);
        Xnorm = 0.5*TensorNorm(X,'fro');
        for n = 1:N
            U0{n} = max(0,randn(Nway(n),para(n)));
            U0{n} = (U0{n}/norm(U0{n},'fro'))*Xnorm^(1/(N+1));
        end
        G0 = randn(para);
        G0 = (G0/TensorNorm(G0,'fro')*Xnorm^(1/(N+1)));
        Rank = size(G0);
    else
        error('Initial parameter is wrong');
    end
    
end  

function [U, Rank] = myRank(X,para,init)

    N = ndims(X);
    Rank = zeros(1,N);
    U = cell(1,N);
    for n = 1:N   
        mat = reshape(shiftdim(X,n-1),size(X,n),[]);
        [U{n},Rank(n)] = TrunSVD(mat, para, init);
    end
    
end

function [U,Rank] = TrunSVD(mat, para, init)

    [U, S, ~] = svd(mat,'econ');
    eigvalue = sort(diag(S),'descend'); 
    if strcmp(init,'truncated')
        Rank = 1; threshold = 1 - para;
        while Rank < length(eigvalue)
            Rank = Rank + 1;
            ratio1 = sum(eigvalue(1:Rank))/sum(eigvalue);
            ratio2 = sum(eigvalue(1:Rank-1))/sum(eigvalue);
            if ratio1 > threshold && ratio2 < threshold
                break;
            end
        end    
     elseif strcmp(init,'tol')
        threshold = para^2 * norm(mat,'fro')^2;
        eigsum = cumsum(eigvalue,'reverse');
        Rank = find(eigsum > threshold, 1, 'last');
    else
        error('Initial parameter is wrong!')
    end
    
    U = U(:,1:Rank);
    U = max(0,U);
    
end