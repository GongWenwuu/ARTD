function psnr = PSNR3D(Image1, Image2, Omega)

    maxP= max(Image1(:));
    Image2 = max(0, Image2);
    Image2 = min(maxP, Image2);

    dim = numel(Image2);
    missing = setdiff(1:dim, Omega);
    num_missing = length(missing);

    Xtemp = Image1 - Image2;
    MSE = norm(Xtemp(missing))^2 / num_missing;
    psnr = 10 * log10(maxP^2 / MSE);

end