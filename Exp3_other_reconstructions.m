%% Create Circles
dims = 200;

gt = phantom(dims-40);
gt = padarray(gt,20*[1,1],0,'both');
dims = size(gt,1);

real_sz = size(gt);

angles = 0:179;
R = ['radonmatrix_size',num2str(dims),'_angles_180.mat'];
if exist(R,'file')
    R = load(R,'R'); 
    R = R.R;
else
    SOLVER = solver(zeros(size(radon(gt,angles)))', angles, angles, 0, 'recon_sz',real_sz);
    R = SOLVER.toMatrix(@(x) reshape(SOLVER.fullR*x(:),size(radon(gt,angles)'))', real_sz);
%     R = radonmtx(dims, angles, size(radon(gt,angles),1));
end
sino_sz = [size(R,1)/numel(angles),numel(angles)];
sino_gt = reshape(R*gt(:),sino_sz);

%% Bad stuff
noise_std = 0.05;
rng(0);
f = sino_gt + (noise_std*max(sino_gt(:)))*randn(sino_sz);
% Limited view
freq = 10; gap = 0;
inpDom = (floor(sino_sz(2)/2)-floor(gap/2)) + (1:gap);
M1 = false(sino_sz);
M1(:,1:freq:end) = true;
M1(:,inpDom) = false;
sub_angles1 = angles(M1(1,:));

% Limited angle
freq = 1; gap = 120;
inpDom = (floor(sino_sz(2)/2)-floor(gap/2)) + (1:gap);
M2 = false(sino_sz);
M2(:,1:freq:end) = true;
M2(:,inpDom) = false;
sub_angles2 = angles(M2(1,:));

clear freq gap inpDom

%% Reconstruction with CVX
E = zeros(2,3);

% FBP
u = iradon(reshape(sino_gt(M2),[],numel(sub_angles2)),sub_angles2);
u = u(2:end-1,2:end-1)*sum(sum(radon(ones(size(gt)),angles)))/sum(R*ones(numel(gt),1));
E(1,1) = norm(u(:)-gt(:));
FBP_res = {u,0};
u = iradon(reshape(f(M1),[],numel(sub_angles1)),sub_angles1);
u = u(2:end-1,2:end-1)*sum(sum(radon(ones(size(gt)),angles)))/sum(R*ones(numel(gt),1));
E(2,1) = norm(u(:)-gt(:));
FBP_res{2} = u;

% ART
u = kaczmarz(R(M2(:),:),reshape(sino_gt(M2),[],1),50);
ART_res = {reshape(u,real_sz),0};
E(1,2) = norm(u(:)-gt(:));
u = kaczmarz(R(M1(:),:),reshape(f(M1),[],1),10);
ART_res{2} = reshape(u,real_sz);
E(2,2) = norm(u(:)-gt(:));

% TV
grad = diffusion_map(gt,1,0);
cvx_begin quiet
    variable u(real_sz)
    Ru = reshape(R*u(:),sino_sz);
    TV = norms(grad.smooth_deriv(u,0,1,1),2,3);
    minimise( sum(TV(:)));
    subject to
        Ru(M2) == sino_gt(M2)
        u(:) >= 0
cvx_end
TV_res = {u,0};
E(1,3) = norm(u(:)-gt(:));
cvx_begin quiet
    variable u(real_sz)
    Ru = reshape(R*u(:),sino_sz);
    TV = norms(grad.smooth_deriv(u,0,1,1),2,3);
    minimise( 1/2*sum_square_abs(Ru(M1) - f(M1)) + 0.0001*sum(TV(:)));
    subject to
        u(:) >= 0
cvx_end
TV_res{2} = u;
E(2,3) = norm(u(:)-gt(:));

disp(num2str(norm(u(:)-gt(:))))


figure;
subplot(2,4,1); imagesc(sino_gt.*M2);title('Sub-sampled Sinogram');
subplot(2,4,2); imagesc(FBP_res{1});title('FBP reconstruction');
subplot(2,4,3); imagesc(ART_res{1});title('SIRT reconstruction');
subplot(2,4,4); imagesc(TV_res{1});title('Optimal TV reconstruction');
subplot(2,4,5); imagesc(f.*M1);title('Noisy Sinogram');
subplot(2,4,6); imagesc(FBP_res{2});title('FBP reconstruction');
subplot(2,4,7); imagesc(ART_res{2});title('SIRT reconstruction');
subplot(2,4,8); imagesc(TV_res{2});title('Optimal TV reconstruction');
for i=1:8
    subplot(2,4,i);
    set(gca,'xtick',[]);set(gca,'ytick',[]);
    if rem(i,4) == 1
        caxis([0,max(sino_gt(:))]);
    else
        axis equal tight; caxis([0 1.2]);
    end
end
clear cvx* k