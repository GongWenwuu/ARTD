function psnr = psnr_index(img1,img2,maxP)
    dim = numel(img1);
    psnr = 10*log10(dim*maxP^2/norm(img1(:)-img2(:))^2);
end