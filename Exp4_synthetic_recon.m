%% Select phantom
dims = 200;

% Shepp-Logan
gt = phantom(dims);
alpha = 3e+2; beta = 3e-5; gamma = 1e-3; L = 200; trust = 4; rho = 1;
sigma = 4; choice_rule = [1e10,1,2]; t=1e-8;

% % Circles:
% gt =  circlecreator(dims*[1,1], [0.5 0.5], 0.3, 2)...
%     - circlecreator(dims*[1,1], [0.5 0.5], 0.2, 2)...
%     + circlecreator(dims*[1,1], [0.5 0.5], 0.15, 2);
% alpha = 3e+3; beta = 3e-5; gamma = 3e-4; L = 200; trust = 2; rho = 1;
% sigma = 8; choice_rule = [1e10,1,2]; t=1e-8;


real_sz = size(gt);

%% Compute (noisy) sinogram
noise_std = 0.05;
angles = 0:179;
sino_gt = radon(gt,angles);
sino_sz = size(sino_gt);

SOLVER = solver(sino_gt', angles, angles, diffusion_map(sino_gt, 0, 0), 'recon_sz', real_sz);
R = @(x) double(reshape(SOLVER.fullR*x(:),sino_sz([2,1]))');
sino_gt = R(gt);

rng(0);
f = sino_gt + (noise_std*max(sino_gt(:)))*randn(sino_sz);

%% Compute mask
freq = 1; gap = 120;
inpDom = (floor(sino_sz(2)/2)-floor(gap/2)) + (1:gap);
M = false(sino_sz); M(:,1:freq:end) = true; M(:,inpDom) = false;
sub_angles = angles(M(1,:));

figure;
subplot(1,3,1);imagesc(sino_gt);title('Fully sampled sinogram')
subplot(1,3,2);imagesc(f);title('Fully sampled sinogram with noise')
subplot(1,3,3);imagesc(M.*f);title('Sub sampled sinogram with noise')
drawnow
clear freq gap inpDom

%% Reconstruction with CVX
clear u_res v_res E H
u_res = cell(0);v_res = cell(0);params = cell(0);
grad = diffusion_map(gt,1,0);
E = zeros(L,1);H = zeros(L,1);



% Perform initial TV inpainting
SOLVER = solver(reshape(f(M),[],numel(sub_angles))', sub_angles, angles, 0,...
    'recon_sz',real_sz,'maxiter',2000,'stepRatio',1);
SOLVER.solve(0,0.0005,1);
u = SOLVER.getResult; TV_res = {u,R(u),norm(u(:)-gt(:))};
u = TV_res{1}; Ru = TV_res{2}; v=Ru;
Norm = @(x) sum(sum(norms(x,2,3)));
D_TV = diffusion_map(Ru', rho(1), sigma(1), 'tmp', choice_rule(1,:));
k=1;
SOLVER = solver_GD(Ru', angles, angles, 0,'sinoW',M'+(1-M')/trust,...
    'recon_sz',real_sz,'maxiter',4000,'stepRatio',10^-2);
SOLVER.warmStart(u);
figure;
while k <= L
    t = min(t,1e-5);
    uk = u; Ruk = Ru; vk = v; dvk=grad.smooth_deriv(vk',0,1,1);Jkvk = D_TV*dvk;
    
    % Perform optimisation for u
    d = D_TV.diff(Ru');
    d = fcthdlop([prod(sino_sz),1],[2*prod(sino_sz),1],@(x)d(dvk,x,false),@(x)d(dvk,x,true));
    SOLVER.solve(alpha,beta, vk',{Jkvk(:)-d*reshape(Ruk',[],1),d},uk,t,true);
    u = SOLVER.getResult; Ru = R(u);

    D_TV = diffusion_map(Ru', rho, sigma, 'tmp', choice_rule);

    % Save and plot u
    u_res{end+1} = u;
    subplot(2,2,3);imagesc(Ru);
    title(['iteration = ', num2str(k)]);
    subplot(2,2,4); imagesc(u); caxis([min(gt(:)),max(gt(:))]);
    E(k) = norm(u(:)-gt(:))/TV_res{3};
    title(['Reconstruction, energy = ', num2str(H(max(k,2)-1))]);axis equal tight;
    drawnow

    % Perform optimisation for v
    v(M) = f(M);
    cvx_begin quiet
        variable v(sino_sz)
        Sino_TV = Norm(D_TV*grad.smooth_deriv(v',0,1,1));
        minimise( sum_square_abs((M(:)+(1-M(:))/trust).*(v(:) - Ru(:)))/2 ...
            + alpha(k)*Sino_TV + (alpha(k)*gamma)*sum_square_abs(v(M)-f(M)));
    cvx_end

    % Save and plot v
    v_res{end+1} = v;
    subplot(2,2,1);plot(H(1:k)); title(['Energy after iteration ', num2str(k)]);xlim([1,max(k,2)]);
    subplot(2,2,2);imagesc(v); title('Inpainting/denoising output'); caxis([0,max(sino_gt(:))]);
    drawnow

    % Compute new energy and decrease step size if necessary
    H(k) = sum_square_abs((M(:)+(1-M(:))/trust).*(Ru(:)-v(:)))/2 ...
        + alpha(k)*sum(Norm(D_TV*grad.smooth_deriv(v',0,1,1))) ...
        + alpha(k)*sum_square_abs(v(M)-f(M))*gamma...
        + beta(k)*sum(sum(norms(grad.smooth_deriv(u,0,1,1),2,3)));
    if H(k)>H(max(1,k-1)); t = t*1.1; end
    k = k + 1;
end
subplot(2,2,3);title('END');disp('END');

clear cvx* k L dims uk Ruk vk Jkvk T s

%% Visualize results
figure;
subplot(2,3,1);imagesc(TV_res{1});axis image; title('TV Reconstruction'); caxis([0,1]);
subplot(2,3,2);imagesc(gt);axis image; title('Ground Truth'); caxis([0,1]);
subplot(2,3,3);imagesc(u);axis image; title('Proposed Reconstruction'); caxis([0,1]);
subplot(2,3,4);imagesc(TV_res{2});caxis([0,max(sino_gt(:))]);
subplot(2,3,5);imagesc(sino_gt);caxis([0,max(sino_gt(:))]);
subplot(2,3,6);imagesc(v);caxis([0,max(sino_gt(:))]);
set(findall(gcf,'-property','FontSize'),'FontSize',26)
set(findall(gcf,'-property','XTick'),'XTick',[])
set(findall(gcf,'-property','YTick'),'YTick',[])