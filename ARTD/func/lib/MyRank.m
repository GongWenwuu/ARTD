function [U, Rank] = MyRank(X,para,init)

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