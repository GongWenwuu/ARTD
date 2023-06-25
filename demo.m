dense_tensor = double(imread('data\lena.bmp'));
R = size(dense_tensor);
Eval_ARTD = zeros(8,12);
CPUTime_ARTD = zeros(1,12);
DLP = [0.1:0.1:0.9,0.93,0.95,0.99];
for MR = 1:length(DLP)
    rng('default')
    sample_ratio = 1- DLP(MR);
    sample_num = round(sample_ratio*numel(dense_tensor));
    fprintf('Sampling OD tensor with %4.1f%% known elements ...... \n',100*sample_ratio);
    % Filter missing positions 
    idx = 1:numel(dense_tensor);
    idx = idx(dense_tensor(:)>0);
    % Artificial missing position
    mask = sort(randperm(length(idx),sample_num));
    arti_miss_idx = idx;  
    arti_miss_idx(mask) = [];  
    arti_miss_mv = dense_tensor(arti_miss_idx);
    Omega = zeros(size(dense_tensor)); Omega(mask) = 1; Omega = boolean(Omega);
    sparse_tensor = Omega.*dense_tensor;
    fprintf('Known elements / total elements: %6d/%6d.\n',sample_num,numel(dense_tensor));
    clear idx 

    t0 = tic;
    Opts = initial_para(300,R,'artd',1,[1,1,1],1e-4,1e-5); Opts.prior = 'stdc'; 
    [est_tensor, ~,~, info] = APG_RTD(dense_tensor, Omega, Opts);
    CPUTime_ARTD(1,MR) = toc(t0);
    rse = TensorNorm(est_tensor - dense_tensor,'fro')/TensorNorm(dense_tensor,'fro');
    nmae = norm(arti_miss_mv-est_tensor(arti_miss_idx),1) / norm(arti_miss_mv,1);
    rmse = sqrt((1/length(arti_miss_mv))*norm(arti_miss_mv-est_tensor(arti_miss_idx),2)^2);  
    [psnr, ssim, fsim, ergas, msam] = MSIQA(dense_tensor, est_tensor);
    Eval_ARTD(1,MR) = psnr; Eval_ARTD(2,MR) = ssim; Eval_ARTD(3,MR) = fsim; Eval_ARTD(4,MR) = ergas; 
    Eval_ARTD(5,MR) = rmse; Eval_ARTD(6,MR) = nmae; Eval_ARTD(7,MR) = msam; Eval_ARTD(8,MR) = rse;

end