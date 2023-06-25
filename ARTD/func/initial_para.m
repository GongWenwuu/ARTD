function para = initial_para(maxit,Rpara,init, beta, flag, tol,epsilon)

para.maxit = maxit;             % maximum iteration number of APG
para.Rpara = Rpara;             % Initial rank
para.init = init;               % Initial Tucker decomposition
para.beta = beta;               % Core shresholding, Low-rank
para.flag = flag;               % Spatiotemporal constriants
para.tol = tol;                 % tolerance parameter for checking convergence
para.epsilon = epsilon;         % rank increment condition

end