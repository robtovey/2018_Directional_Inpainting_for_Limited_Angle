function [m,bestIdx] = visualize_results_inpainting(u_res,alpha,f,exact,mask,nx,ny)

for k = 1 : length(alpha)
    % best recon based on MSE for inpainting domain:
    mse_vector(k) = norm(exact(mask(:))-u_res{k}(mask(:)),2)/norm(exact(mask(:)),2);
end
figure;
plot(alpha,mse_vector,'--o'); xlabel('alpha'); ylabel('|u-urec|/|u|');
[m,bestIdx] = min(mse_vector(:));
figure;
subplot(2,2,1); imagesc(exact); colormap(gray); colorbar; title('Original'); axis image;
subplot(2,2,2); imagesc(reshape(f,nx,ny)); colormap(gray); colorbar; title('f'); axis image;
subplot(2,2,3); imagesc(reshape(u_res{bestIdx},nx,ny)); colormap(gray); colorbar; title(['result, alpha=',num2str(alpha(bestIdx))]); axis image;
subplot(2,2,4); imagesc(reshape(exact(:)-u_res{bestIdx},nx,ny)); colormap(gray); colorbar; title('difference'); axis image;
