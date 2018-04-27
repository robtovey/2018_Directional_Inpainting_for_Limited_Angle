%% Select reconstruction
f = zeros(1024,1024,46);
for i=1:46
    f(:,:,i) = double(imread('p10arot90_hdr0_5.tif',i));
end
f = squeeze(f(500,:,:));
f = f(1:6:end,:);f = f*100/sum(f(:));
f(1:20,:) = 0;
sub_angles = -68:3:67;


% % Fully sampled
% M = [1,0]; alpha=1e-4; beta=3e-5; gamma=1; L=100; trust=2; rho=1; sigma=0;
% choice_rule=[1e6,1e2,2]; t=1e-7;

% Limited angle
M = [1,15]; alpha=1e-4; beta=3e-5; gamma=10; L=100; trust=2; rho=1; sigma=0;
choice_rule=[1e6,1e2,2]; t=1e-7;


%%
angles = -90:89;
real_sz = size(iradon(f,sub_angles));
sino_sz = size(radon(zeros(real_sz),angles));
f = padarray(f,[2,0],0,'post');

SOLVER = solver(zeros(sino_sz)', angles, -90:269, 0, 'recon_sz', real_sz);
% R = SOLVER.toMatrix(@(x) reshape(SOLVER.fullR*x(:),sino_sz([2,1]))', real_sz);
R = @(x) double(reshape(SOLVER.subR*x(:),sino_sz([2,1]))');


%% Compute mask
M = false(sino_sz);
M(:,sub_angles+91) = true;
MM = M; MM(:,91+(sub_angles(1):sub_angles(end))) = true;

%% Reconstruction with CVX
clear u_res v_res H
u_res = cell(0);v_res = cell(0);params = cell(0);
grad = diffusion_map(f,1,0);
H = zeros(L,1);


% Perform initial TV inpainting
SOLVER = solver(f', sub_angles, -90:269, 0,...
    'recon_sz',real_sz,'maxiter',5000,'stepRatio',1);
SOLVER.solve(0,1e-5,0);
u = SOLVER.getResult; TV_res = {u,R(u)};
u = TV_res{1}; Ru = TV_res{2}; v=Ru;

% Estimate Lipschitz constant
Norm = @(x) sum(sum(norms(x,2,3)));
to_full = @(Ru) [Ru,flipud(Ru)];
fullR = solver(to_full(Ru)', -90:269, -90:269, 0, 'recon_sz', real_sz); fullR = fullR.fullR;
D_TV = diffusion_map(to_full(Ru)', rho, sigma, 'tmp2', choice_rule);
k=1;
if trust == 1
    SOLVER = solver_GD(Ru', angles, angles, 0,...
        'recon_sz',real_sz,'maxiter',4000,'stepRatio',1e-2);
else
    SOLVER = solver_GD(Ru', angles, -90:269, 0,'sinoW',to_full(M+(1-M)/trust)',...
        'recon_sz',real_sz,'maxiter',4000,'stepRatio',10^-2);
end
SOLVER.warmStart(u);
while k <= L
    uk = u; Ruk = to_full(Ru); vk = v; dvk=grad.smooth_deriv(to_full(vk)',0,1,1);Jkvk = D_TV*dvk;
    vk(M) = f(:);
    
    % Perform optimisation for u
    d = D_TV.diff(to_full(Ru)');
    d = fcthdlop([2*prod(sino_sz),1],[4*prod(sino_sz),1],@(x)d(dvk,x,false),@(x)d(dvk,x,true));
    SOLVER.solve(alpha,beta, to_full(vk)',{Jkvk(:)-d*reshape(Ruk',[],1),d},uk,t,true);
    u = SOLVER.getResult; Ru = R(u);
    
    D_TV = diffusion_map(to_full(Ru)', rho, sigma, 'tmp2', choice_rule);

    % Save and plot u
    u_res{end+1} = u;
    subplot(2,2,3);imagesc(Ru);
    title(['iteration = ', num2str(k)]);
    subplot(2,2,4); imagesc(u);caxis([0,0.145])
    title(['Reconstruction, energy = ', num2str(H(max(k,2)-1))]);axis equal tight;
    drawnow

    % Perform optimisation for v
    cvx_begin quiet
        variable v(sino_sz)
        Sino_TV = Norm(D_TV*grad.smooth_deriv(to_full(v)',0,1,1));
        minimise( sum_square_abs(Ru(~M)-v(~M))/(2*trust^2) ...
            + alpha(k)*Sino_TV+(gamma)*sum_square_abs(v(M)-f(:)));
    cvx_end
    
    % Save and plot v
    v_res{end+1} = v;
    subplot(2,2,1);plot(H(1:k)); title(['Energy after iteration ', num2str(k)]);xlim([1,max(k,2)]);
    subplot(2,2,2);imagesc(v); title('Inpainting/denoising output'); caxis([0,max(f(:))]);
    drawnow

    % Compute new energy and decrease step size if necessary
    H(k) = sum_square_abs((M(:)+(1-M(:))/trust).*(Ru(:)-v(:)))/2 ...
        + alpha(k)*sum(Norm(D_TV*grad.smooth_deriv(to_full(v)',0,1,1))) ...
        + sum_square_abs(v(M)-f(:))*gamma...
        + beta(k)*sum(sum(norms(grad.smooth_deriv(u,0,1,1),2,3)));
    if H(k)>H(max(1,k-1)); t = t*1.1; end
    k = k + 1;
end
subplot(2,2,3);title('END');disp('END');

clear cvx* k L dims uk Ruk vk Jkvk T s

%% Visualize results
figure;
subplot(1,2,1);imagesc(TV_res{1});axis image; title('TV Reconstruction, Full Data'); caxis([0,0.145]);
subplot(1,2,2);imagesc(u);axis image; title('Proposed Reconstruction, Full Data'); caxis([0,0.145]);
set(findall(gcf,'-property','FontSize'),'FontSize',26)
set(findall(gcf,'-property','XTick'),'XTick',[])
set(findall(gcf,'-property','YTick'),'YTick',[])

thresh=0.055;figure;
subplot(1,2,1);imagesc(TV_res{1}>thresh); title('TV Reconstruction, Full Data')
grid;set(gca,'GridColor','red');set(gca,'LineWidth',1);set(gca,'GridAlpha',1);
subplot(1,2,2);imagesc(u>thresh); title('Proposed Reconstruction, Full Data');
grid;set(gca,'GridColor','red');set(gca,'LineWidth',1);set(gca,'GridAlpha',1);
set(findall(gcf,'-property','FontSize'),'FontSize',26)
