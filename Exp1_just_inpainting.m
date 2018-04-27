%% General
res = 512;
width = round(0.2*512);

D = diffusion_map(zeros(512),1,0);

split = floor((res-width)/2) + [0,width-1];
dsplit = round(linspace(1,res,8)); 
dsplit = [dsplit(2),dsplit(4); dsplit(5),dsplit(7)];

mask = true(res);
mask(floor(res/4):floor(3*res/4),floor(res/4):floor(3*res/4)) = false;
% mask(floor(2.5*res/6):floor(3.5*res/6),floor(res/4):floor(3*res/4)) = false;

%% Rectangle
gt = zeros(res);
gt(split(1):split(2),:) = 1;

c1 = ones(res);
c1(split(1)+[-1,0,1],:) = 0;c1(split(2)+[-1,0,1],:) = 0;
c2 = ones(res);

cvx_begin quiet
    variable u1([res,res])
    Du = reshape(D.smooth_deriv(u1,0,1,1),res*res,2);
    minimise sum(norms([1.*Du(:,1),1.*Du(:,2)],2,2))
    subject to 
        u1(mask) == gt(mask)
cvx_end
cvx_begin quiet
    variable u2([res,res])
    Du = reshape(D.smooth_deriv(u2,0,1,1),res*res,2);
    minimise sum(norms([c1(:).*Du(:,1),c2(:).*Du(:,2)],2,2))
    subject to 
        u2(mask) == gt(mask)
cvx_end
clear cvx*

figure;
subplot(2,2,1);imagesc((gt+1).*mask-1);title('Input');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,2);imagesc(u1);title('TV Output');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,3);imagesc(c1);caxis([0,1]); title('c_1');set(gca,'fontsize', 18);
subplot(2,2,4);imagesc(u2);title('DTV Output');set(gca,'fontsize', 18);caxis([-1,1]);

%% Noisy Rectangle
gt = zeros(res);
gt(split(1):split(2),:) = 1;
rng(1);noise = 2*randn(size(gt));

c1 = ones(res);
c1(split(1)+(-2:2),:) = 0;c1(split(2)+(-2:2),:) = 0;
c2 = ones(res);
c1 = imgaussfilt(c1,2);c2 = imgaussfilt(c2,2);

cvx_begin quiet
    variable u1([res,res])
    Du = norms(D.smooth_deriv(u1,0,1,1),2,3);
%     minimise sum_square_abs(u1(:)-gt(:)-noise(:)) + 10*sum(Du(:))
    minimise sum(Du(:))
    subject to
        sum_square_abs(u1(:)-gt(:)-noise(:)) <= sum_square_abs(noise(:))
cvx_end
cvx_begin quiet
    variable u2([res,res])
    Du = norms(cat(3,c1,c2).*D.smooth_deriv(u2,0,1,1),2,3);
%     minimise sum_square_abs(u2(:)-gt(:)-noise(:)) + 10*sum(Du(:))
    minimise sum(Du(:))
    subject to
        sum_square_abs(u2(:)-gt(:)-noise(:)) <= sum_square_abs(noise(:))
cvx_end
clear cvx*

figure;
subplot(2,2,1);imagesc(gt+noise);title('Input');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,2);imagesc(u1);title('TV Output');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,3);imagesc(c1);caxis([0,1]); title('c_1');set(gca,'fontsize', 18);
subplot(2,2,4);imagesc(u2);title('DTV Output');set(gca,'fontsize', 18);caxis([-1,1]);

%% Parallel Rectangles
gt = zeros(res);
gt(dsplit(1,1):dsplit(1,2),:) = 1;
gt(dsplit(2,1):dsplit(2,2),:) = 1;

c1 = ones(res);
c1(dsplit(1,1)+[-1,0,1],:) = 0;c1(dsplit(1,2)+[-1,0,1],:) = 0;
c1(dsplit(2,1)+[-1,0,1],:) = 0;c1(dsplit(2,2)+[-1,0,1],:) = 0;
c2 = ones(res);
% c1 = imgaussfilt(c1,2);c2 = imgaussfilt(c2,2);

cvx_begin quiet
    variable u1([res,res])
    Du = norms(D.smooth_deriv(u1,0,1,1),2,3);
    minimise sum(Du(:))
    subject to 
        u1(mask) == gt(mask)
cvx_end
cvx_begin quiet
    variable u2([res,res])
    Du = reshape(D.smooth_deriv(u2,0,1,1),res*res,2);
    minimise sum(norms([c1(:).*Du(:,1),c2(:).*Du(:,2)],2,2))
    subject to 
        u2(mask) == gt(mask)
cvx_end
clear cvx* tmpD

figure;
subplot(2,2,1);imagesc((gt+1).*mask-1);title('Input');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,2);imagesc(u1);title('TV Output');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,3);imagesc(c1);caxis([0,1]); title('c_1');set(gca,'fontsize', 18);
subplot(2,2,4);imagesc(u2);title('DTV Output');set(gca,'fontsize', 18);caxis([-1,1]);

%% Intersecting Rectangles
gt = zeros(res);
gt(split(1):split(2),:) = 1;
gt(:,split(1):split(2)) = 1;

c1 = .7*ones(res);
c1(split(1)+[-1,0,1],:) = 0;c1(split(2)+[-1,0,1],:) = 0;
c1(:,split(1)+[-1,0,1],:) = 0;c1(:,split(2)+[-1,0,1]) = 0;
c2 = ones(res);
c1 = imgaussfilt(c1,2);c2 = imgaussfilt(c2,2);

tmpD = diffusion_map(gt, 16, 16, 'tmp', [1,1,2]);
tmpD.c1_val = c1.^2;tmpD.c2_val = c2.^2;
cvx_begin quiet
    variable u1([res,res])
    Du = norms(D.smooth_deriv(u1,0,1,1),2,3);
    minimise sum(Du(:))
    subject to 
        u1(mask) == gt(mask)
cvx_end
cvx_begin quiet
    variable u2([res,res])
    Du = reshape(tmpD*D.smooth_deriv(u2,0,1,1),res*res,2);
    minimise sum(norms(Du,2,2))
    subject to 
        u2(mask) == gt(mask)
cvx_end
clear cvx* tmpD

figure;
subplot(2,2,1);imagesc((gt+1).*mask-1);title('Input');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,2);imagesc(u1);title('TV Output');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,3);imagesc(c1);caxis([0,1]); title('c_1');set(gca,'fontsize', 18);
subplot(2,2,4);imagesc(u2);title('DTV Output');set(gca,'fontsize', 18);caxis([-1,1]);

%% Gradient Rectangle
gt = zeros(res);
gt(split(1):split(2),:) = repmat(linspace(0,1,res),width,1);

c1 = ones(res);
c1(split(1)+[-1,0,1],:) = 0;c1(split(2)+[-1,0,1],:) = 0;
c2 = ones(res);
% c1 = imgaussfilt(c1,2);c2 = imgaussfilt(c2,2);

cvx_begin quiet
    variable u1([res,res])
    Du = norms(D.smooth_deriv(u1,0,1,1),2,3);
    minimise sum(Du(:))
    subject to 
        u1(mask) == gt(mask)
cvx_end
cvx_begin quiet
    variable u2([res,res])
    Du = reshape(D.smooth_deriv(u2,0,1,1),res*res,2);
    minimise sum(norms([c1(:).*Du(:,1),c2(:).*Du(:,2)],2,2))
    subject to 
        u2(mask) == gt(mask)
cvx_end
clear cvx*

figure;
subplot(2,2,1);imagesc((gt+1).*mask-1);title('Input');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,2);imagesc(u1);title('TV Output');set(gca,'fontsize', 18);caxis([-1,1]);
subplot(2,2,3);imagesc(c1);caxis([0,1]); title('c_1');set(gca,'fontsize', 18);
subplot(2,2,4);imagesc(u2);title('DTV Output');set(gca,'fontsize', 18);caxis([-1,1]);
