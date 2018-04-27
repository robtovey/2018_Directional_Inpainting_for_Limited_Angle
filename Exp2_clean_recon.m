%% Settings
dims = 200;
angles = 0:179;
real_sz = dims*[1,1];
sino_sz = size(radon(zeros(real_sz),angles));


if any(strcmp(num2str(dims),{'200','64'}))
    R = load(['radonmatrix_size',num2str(dims),'_angles_180.mat'],'R');
    R = R.R;
else
    SOLVER = solver(zeros(sino_sz([2,1])), angles, angles, 0, 'recon_sz', real_sz);
    R = @(x) double(reshape(SOLVER.fullR*x(:),sino_sz([2,1]))');
%     R = SOLVER.toMatrix(@(x) reshape(SOLVER.fullR*x(:),sino_sz([2,1]))', real_sz);
end

freq = 1; gap = 120;
inpDom = (floor(sino_sz(2)/2)-floor(gap/2)) + (1:gap);
M = false(sino_sz);
M(:,1:freq:end) = true;
M(:,inpDom) = false;
sub_angles = angles(M(1,:));

clear freq gap inpDom

%% Select phantom:
% Shepp-Logan
gt = phantom(dims);
% % Circles
% gt =  circlecreator(dims*[1,1], [0.5 0.5], 0.3, 2)...
%     - circlecreator(dims*[1,1], [0.5 0.5], 0.2, 2)...
%     + circlecreator(dims*[1,1], [0.5 0.5], 0.15, 2);
% % Squares
% gt =  circlecreator(dims*[1,1], [0.5 0.5], 0.3, inf)...
%     - circlecreator(dims*[1,1], [0.5 0.5], 0.2, inf)...
%     + circlecreator(dims*[1,1], [0.5 0.5], 0.15, inf);

%% Do Reconstruction
if isnumeric(R)
    sino_gt = reshape(R*gt(:),sino_sz);

    grad = diffusion_map(gt,1,0);
    cvx_begin
        variable u(real_sz)
        Ru = reshape(R*u(:),sino_sz);
        TV = norms(grad.smooth_deriv(u,0,1,1),2,3);
        minimise( sum(TV(:)));
        subject to
            u(:) >= 0
            Ru(M) == sino_gt(M)
    cvx_end

    clear cvx* TV grad
else
    sino_gt = R(gt);

    SOLVER = solver(reshape(sino_gt(M),[],numel(sub_angles))', sub_angles, angles, 0,...
        'recon_sz',real_sz,'maxiter',100000,'stepRatio',10);
    SOLVER.warmStart(gt);
    SOLVER.solve(0,0,false);
    u = SOLVER.getResult;
    Ru = R(u);
end

figure;
subplot(2,3,1);imagesc(iradon(reshape(sino_gt(M),sino_sz(1),[]),sub_angles)); title('FBP');
subplot(2,3,2);imagesc(gt); title('GT');
subplot(2,3,3);imagesc(u); title('TV recon');
subplot(2,3,4);imagesc(M.*sino_gt); title('Sub-sampled Sinogram');
subplot(2,3,5);imagesc(sino_gt); title('Fully sampled Sinogram');
subplot(2,3,6);imagesc(Ru); title('TV recon');