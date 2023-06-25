function PSNR = psnrpsnr(Image1,Image2,Omega)
% Compute the peak SNR given two images
    if size(Image1,3) > 1
        Image1 = rgb2gray(Image1);
    end
    if size(Image2,3) > 1
        Image2 = rgb2gray(Image2);
    end
    Image1 = double(Image1);Image2 = double(Image2);
    MSE = TensorNorm(Image2-Image1,'fro').^2/length(nonzeros(~Omega));
    MAXi = max(abs(Image2(:)))^2;

    PSNR = 	10*log10(MAXi/MSE);

return;