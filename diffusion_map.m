classdef diffusion_map < handle
% Here we calculate diffusion tensors for enhancing edges in images.
% We take $$ M(f) = (\nabla f_\rho \nabla f_\rho^T)_\sigma$$
% where subscript denotes convolution with Gaussian kernel of that std.
% Next we decompose $$ M = \lambda_1 e_1\otimes e_1 + \lambda_2 e_2\otimes
%   e_2 \text{ such that } e_1 \parallel \nabla f_\rho$$
% Finally, we construct our tensor:
% $$ J = c_1(\lambda_1,\lambda_2) e_1\otimes e_1 + c_2(\lambda_1,\lambda_2)
%   e_2\otimes e_2$$
% where c_1 is large and c_2 is small to enhance edges.
% 
% In practice we have
% J = diffusion_map(f, rho, sigma, choiceRule, Parameters)
% Inputs:
%   f: the (2D) image to form $M$
%   rho: smoothing parameter before taking gradient
%   sigma: smoothing parameter after taking gradient
%   choiceRule, Parameters: allows you to specify a default form of c1/c2.
%       choiceRule='symmetric': assume Parameters = [g,m]
%               c1 = (sqrt(\lambda_1+g*\lambda_2)/\lambda_1)^m
%               c2 = (sqrt(\lambda_1+g*\lambda_2)/\lambda_2)^m
%               Recommend g = 1e-3, m = 0.5
%       choiceRule='isotropic': assume Parameters = [g,k]
%               c1 = g + 1/sqrt(1+\lambda_1^2/k^2)
%               c2 = 1+g
%               Recommend g = 0, k = 3
%       choiceRule='exponential': assume Parameters = [g,m,k]
%               c1 = 1+g - \exp(-k^m/(\lambda_1-\lambda_2)^m)
%               c2 = 1+g
%               Recommend g = 0, m=4, k = 3
%       choiceRule='coherence': assume Parameters = [g,m,k]
%               c1 = g
%               c2 = g + (1-g)\exp(-k/(\lambda_1-\lambda_2)^{2m})
%               Recommend g = 0.001, m=1, k = 1
%       choiceRule='threshold': assume Parameters = [g,t]
%               c1 = g
%               c2 = 1 if \lambda_1>t, g else
%       choiceRule='Esteller': assume Parameters=t
%               c1 = (max(t,sqrt(\lambda_1+\lambda_2))/\lambda_1)
%               c2 = (max(t,sqrt(\lambda_1+\lambda_2))/\lambda_2)
%               Note almost same as 'symmetric'
%       otherwise assume Parameters = {c1,c2} where c1 =
%               c1(\lambda_1,\lambda_2) etc.
% The above methods can be found with examples in "Anisotropic Diffusion
% in Image Processing" by Joachim Weickert.
    properties(Access=public)
        f
        sigma
        rho
        choiceRule
        Parameters
    end
    properties(Access=public)
        C1
        C2
        DC1
        DC2
        X1
        X2
        h_rho
        h_sigma
        e1_val
        e2_val
        c1_val
        c2_val
    end
    
    methods(Access=public)
        function obj = diffusion_map(f, rho, sigma, choiceRule, Parameters)
            obj.rho = rho;
            obj.sigma = sigma;

            if nargin < 5
                choiceRule = 'coherence';
                Parameters = [0,2,0.1];
            end
            obj.f = f;
            obj.set_smoothing(rho, sigma);
            obj.set_choiceRule(choiceRule, Parameters);
        end
        
        function set_image(obj, f)
            obj.f = full(f);
            obj.reset;
        end
        function set_smoothing(obj, rho, sigma)
            obj.rho = rho;
            obj.sigma = sigma;

            sz = size(obj.f);
            x1 = [0:floor(sz(1)/2), 1-ceil(sz(1)/2):-1];
            x2 = [0:floor(sz(2)/2), 1-ceil(sz(2)/2):-1];
            [x1,x2] = ndgrid(x1,x2);
            obj.X1 = x1; obj.X2 = x2;
            if rho == 0
                obj.h_rho = 1;
            else
                obj.h_rho = exp(-(x1.*x1+x2.*x2)/(2*rho^2));
                obj.h_rho = fft2(obj.h_rho/sum(obj.h_rho(:)));
            end
            if sigma == 0
                obj.h_sigma = 1;
            else
                obj.h_sigma = exp(-(x1.*x1+x2.*x2)/(2*sigma^2));
                obj.h_sigma = fft2(obj.h_sigma/sum(obj.h_sigma(:)));
            end
            obj.reset;
        end
        function set_choiceRule(obj, choiceRule, Parameters)
            switch choiceRule
                case 'symmetric'
                    g = Parameters(1);
                    m = Parameters(2);
                    c1 = @(lambda1, lambda2) (sqrt(lambda1+lambda2.*g)./max(eps,lambda1)).^m;
                    c2 = @(lambda1, lambda2) (sqrt(lambda1+lambda2.*g)./max(eps,lambda2)).^m;
                case 'isotropic'
                    g = Parameters(1);
                    k = Parameters(2);
                    c1 = @(lambda1, lambda2) g + 1./sqrt(1+lambda1.^2/(max(lambda1(:))*k)^2);
                    c2 = @(lambda1, lambda2) 1+g;
                case 'exponential'
                    g = Parameters(1);
                    m = Parameters(2);
                    k = Parameters(3);
                    Cm = fzero(strcat('1-exp(-x)-x*exp(-x)*',num2str(m)),[1e-10, sqrt(3*m+9/16-3)-1/2]);
                    
                    c1 = @(Delta, Sigma) 1+g-exp(-k*Cm*((max(Delta(:)))./Delta).^m);
                    c2 = @(Delta, Sigma) 1+g;

                    dc1 = @(Delta, Sigma) cat(3,-exp(-k*Cm*(max(Delta(:))./Delta).^m).*(k*Cm*m*(max(Delta(:))./Delta).^(m+1)),zeros(size(Sigma)));                    
                    dc2 = @(Delta, Sigma) cat(3,zeros(size(Delta)),zeros(size(Sigma)));
                case 'coherence'
                    g = Parameters(1);
                    m = Parameters(2);
                    k = Parameters(3);
                    c1 = @(Delta, Sigma) g*ones(size(Delta));
                    c2 = @(Delta, Sigma) g+(1-g)*exp(-(k./Delta).^(2*m));
                    
                    dc1 = @(Delta, Sigma) cat(3,zeros(size(Delta)),zeros(size(Sigma)));
                    dc2 = @(Delta, Sigma) cat(3,(1-g)*exp(-k./Delta.^(2*m)).*(k*2*m*(1./Delta).^(2*m+1)),zeros(size(Sigma)));
                case 'tmp'
                    g = Parameters(1);
                    t = Parameters(2);
                    k = Parameters(3);
                    
                    if t == inf
                    c1 = @(Delta, Sigma) 1e-6+(1+(g*Delta).^k).^(-.5/k);
                    c2 = @(Delta, Sigma) 1e-6+1;
                    
                    dc1 = @(Delta,Sigma) cat(3,(-g/2)*tanh(t*Sigma).*((g*Delta).^(k-1)./((1+(g*Delta).^k).^(1+.5/k))), ...
                        zeros(size(Delta)));
                    dc2 = @(Delta,Sigma) cat(3,zeros(size(Delta)),zeros(size(Delta)));
                    else
                        
                    c1 = @(Delta, Sigma) 1e-6+tanh(t*Sigma).*(1+(g*Delta).^k).^(-.5/k);
                    c2 = @(Delta, Sigma) 1e-6+tanh(t*Sigma);

                    dc1 = @(Delta,Sigma) cat(3,(-g/2)*tanh(t*Sigma).*((g*Delta).^(k-1)./((1+(g*Delta).^k).^(1+.5/k))), ...
                        (t*sech(t*Sigma).^2).*(1+(g*Delta).^k).^(-.5/k));
                    dc2 = @(Delta,Sigma) cat(3,zeros(size(Delta)),t*sech(t*Sigma).^2);
                    end
                case 'tmp2'
                    g = Parameters(1);
                    t = Parameters(2);
                    k = Parameters(3);
                    
                    c1 = @(Delta, Sigma) 1e-5+(1-tanh(t*Sigma)).*(1+(g*Delta).^k).^(-.5/k);
                    c2 = @(Delta, Sigma) 1e-5+(1-tanh(t*Sigma));

                    dc1 = @(Delta,Sigma) cat(3,(-g/2)*tanh(t*Sigma).*((g*Delta).^(k-1)./((1+(g*Delta).^k).^(1+.5/k))), ...
                        (t*sech(t*Sigma).^2).*(1+(g*Delta).^k).^(-.5/k));
                    dc2 = @(Delta,Sigma) cat(3,zeros(size(Delta)),t*sech(t*Sigma).^2);
                case 'Esteller'
                    t = Parameters(1);
                    c1 = @(Delta, Sigma) (max(t,sqrt(Sigma))./( Sigma+Delta + (Sigma+Delta<1e-6)*1e-6));
                    c2 = @(Delta, Sigma) (max(t,sqrt(Sigma))./( Sigma-Delta + (Sigma-Delta<1e-6)*1e-6));
                    
                    dc1 = @(Delta,Sigma) cat(3,-(c1(Delta,Sigma)>t).*sqrt(Sigma)./((Sigma+Delta).^2+eps),...
                        (c1(Delta,Sigma)>t).*(Delta-Sigma)./(2*sqrt(Sigma).*(Sigma+Delta).^2+eps));
                    dc2 = @(Delta,Sigma) cat(3,(c1(Delta,Sigma)>t).*sqrt(Sigma)./((Sigma-Delta).^2+eps),...
                        -(c1(Delta,Sigma)>t).*(Delta+Sigma)./(2*sqrt(Sigma).*(Sigma-Delta).^2+eps));
                otherwise
                    [c1,c2,dc1,dc2] = deal(Parameters{:});
            end
            obj.choiceRule = choiceRule;
            obj.Parameters = Parameters;
            obj.C1 = c1; obj.C2 = c2;
            obj.DC1 = dc1; obj.DC2 = dc2;
            obj.reset;
        end
        function reset(obj)
            if isempty(obj.C1)
                return
            end
            % Smooth image then differentiate
            Df = obj.smooth_deriv(obj.f, 'rho', 1,1);
            % Calculate M = DfDf^T then smooth
            M = cat(3, Df(:,:,1).*Df(:,:,1),...
                       Df(:,:,1).*Df(:,:,2),...
                       Df(:,:,2).*Df(:,:,2));
            M = obj.smooth_deriv(M, 'sigma', 0);
            % Decompose matrix
            [l1, l2, e1, e2] = obj.eig_decomp(M);

            % Compute altered eigenvalues and store all
            obj.e1_val = full(e1); obj.e2_val = full(e2);
%             figure;subplot(1,2,1);imagesc(l1);subplot(1,2,2);imagesc(l2);
            obj.c1_val = full(obj.C1(l1-l2,l1+l2));
            obj.c2_val = full(obj.C2(l1-l2,l1+l2));
%             scale = max(max(obj.c1_val(:)),max(obj.c2_val(:)));
%             obj.c1_val = obj.c1_val/scale;
%             obj.c2_val = obj.c2_val/scale;
        end
        function plot(obj)
%             visualize_orientation(obj.c1_val,obj.c2_val,obj.e2_val,obj.f);
            figure;
%             tmp = [min(min(obj.c1_val(:)),min(obj.c2_val(:))), max(max(obj.c1_val(:)),max(obj.c2_val(:)))];
            subplot(1,2,1);imagesc(obj.c1_val);title('Penalty diffusing accross lines');%caxis(tmp);
            set(gca,'XTick',[]);set(gca,'YTick',[]);
            subplot(1,2,2);imagesc(obj.c2_val);title('Penalty diffusing along lines');%caxis(tmp);
            set(gca,'XTick',[]);set(gca,'YTick',[]);


            figure; imagesc(obj.f); colormap(gray); hold on;
            % for plotting purposes x and y direction have to be changed (Matlab's row/col)
            sz = size(obj.f); n = floor((0:49)*sz(1)/50)+1;m = floor((0:49)*sz(2)/50)+1;
            quiver(m,n,obj.e2_val(n,m,2),obj.e2_val(n,m,1)); hold off;
        end
        
        function sz = size(obj, arg)
            if nargin == 1
                sz = size(obj.f);
            else
                sz = size(obj.f,arg);
            end
        end
        function n = numel(obj)
            n = numel(obj.f);
        end
        
        function Du = smooth_deriv(obj, u, std, deriv, order)
            % Convolution first then differential
            if nargin < 5
                order = 2;
            end
            if std ~= 0
                if strcmp(std,'rho')
                    h = obj.h_rho;
                elseif strcmp(std,'sigma')
                    h = obj.h_sigma;
                else
                    h = exp(-(obj.X1.*obj.X1+obj.X2.*obj.X2)/(2*std^2));
                    h = fft2(h/sum(h(:)));
                end
                if numel(h) > 1
                    if isa(u,'cvx')
                        sz = [size(u) 1];
                        F = load(['fft2matrix_size',num2str(sz(1)),'x',num2str(sz(2))],'F');F = F.F;
                        u = reshape(...
                                real(F'*(...
                                    (F*reshape(u,[],sz(3))...
                                     ).*repmat(h(:),[1,sz(3)]))...
                                 )...
                             ,sz);
                    else
                        u = real(ifft2(fft2(u).*repmat(h,[1 1 size(u,3)])));
                    end
                end
            end
            Du = grad(u, deriv, order);
        end
        function [l1,l2,e1,e2] = eig_decomp(~,M)
            % If M(2) = 0 then M is diagonal
            special = (M(:,:,2) == 0);
            inv_b = 1./M(:,:,2);
            inv_b(special) = 0;
            special = special & (M(:,:,1)<M(:,:,3));
            count = nnz(special);
            
            diff = 0.5*(M(:,:,1) - M(:,:,3));
            delta = sqrt(diff.*diff + M(:,:,2).*M(:,:,2));
            
            % l1,l2 = (M(1)+M(3))/2 \pm delta
            l1 = 0.5*(M(:,:,1)+M(:,:,3)) + delta;
            l2 = 0.5*(M(:,:,1)+M(:,:,3)) - delta;
            
            
            % e_1 \parallel [1, (delta-diff)/M(2))]
            % e_2 \parallel [-e1(2), 1]
            e1 = (delta-diff).*inv_b; 
            norm = 1./sqrt(1 + e1.^2);
            e1 = cat(3,norm,norm.*e1);
            e2 = cat(3,-e1(:,:,2),e1(:,:,1));
            
            if count > 0
                % deal with diagonal case
                e1 = reshape(e1, [], 2); e2 = reshape(e2, [], 2);
                e1(special(:),:) = repmat([0,1],count,1);
                e2(special(:),:) = repmat([1,0],count,1);
                e1 = reshape(e1, size(l1,1),[], 2); e2 = reshape(e2, size(l1,1),[], 2);
            end
            
            % Remove small precision errors
            special = l1<1e-10;
            l1(special) = eps;
            l2(special) = eps;
            
%             M1 = l1.*e1(:,:,1).^2+l2.*e2(:,:,1).^2;
%             M2 = l1.*e1(:,:,1).*e1(:,:,2)+l2.*e2(:,:,1).*e2(:,:,2);
%             M3 = l1.*e1(:,:,2).^2+l2.*e2(:,:,2).^2;
%             figure;
%             subplot(1,3,1);pcolor(M1);shading flat; colorbar;
%             subplot(1,3,2);pcolor(M2);shading flat; colorbar;
%             subplot(1,3,3);pcolor(M3);shading flat; colorbar;
%             figure;
%             subplot(1,3,1);pcolor(M(:,:,1));shading flat; colorbar;
%             subplot(1,3,2);pcolor(M(:,:,2));shading flat; colorbar;
%             subplot(1,3,3);pcolor(M(:,:,3));shading flat; colorbar;
        end
        function V = mtimes(J, Du)
            Du = reshape(Du,size(J.e1_val));
%             V = cat(3,...
%                 J.c1_val.*(J.e1_val(:,:,1).*Du(:,:,1) + J.e1_val(:,:,2).*Du(:,:,2)),...
%                 J.c2_val.*(J.e2_val(:,:,1).*Du(:,:,1) + J.e2_val(:,:,2).*Du(:,:,2)));
            V = cat(3,...
                J.c1_val.*(J.e1_val(:,:,1).*J.e1_val(:,:,1).*Du(:,:,1) + J.e1_val(:,:,1).*J.e1_val(:,:,2).*Du(:,:,2)) + J.c2_val.*(J.e2_val(:,:,1).*J.e2_val(:,:,1).*Du(:,:,1) + J.e2_val(:,:,1).*J.e2_val(:,:,2).*Du(:,:,2)),...
                J.c1_val.*(J.e1_val(:,:,2).*J.e1_val(:,:,1).*Du(:,:,1) + J.e1_val(:,:,2).*J.e1_val(:,:,2).*Du(:,:,2)) + J.c2_val.*(J.e2_val(:,:,2).*J.e2_val(:,:,1).*Du(:,:,1) + J.e2_val(:,:,2).*J.e2_val(:,:,2).*Du(:,:,2)));
        end
        
        function [DJM,J] = diff(obj, u)
%  obj(u) = J(M(u)) where:
%       M(u) = (\nabla u_\rho \nabla u_\rho^T)_\sigma
%       J(\lambda_1d^1d^1' + \lambda_2d^2d^2') = c_1(\lambda)e^1d^1' + c_2(\lambda)e^2d^2'
        sz = [size(u,1),size(u,2)];
        tol = 1e-10;
        
        Du_rho = obj.smooth_deriv(u,'rho',1,1);
        M = cat(3, Du_rho(:,:,1).*Du_rho(:,:,1),...
                   Du_rho(:,:,1).*Du_rho(:,:,2),...
                   Du_rho(:,:,2).*Du_rho(:,:,2));
        M = obj.smooth_deriv(M, 'sigma', 0);
        
%         nabM = @(D) DM(obj, Du_rho, D);
        
        M11mM22 = M(:,:,1)-M(:,:,3);
        Sigma = M(:,:,1)+M(:,:,3);
        Delta = sqrt(M11mM22.^2+4*M(:,:,2).^2);
        d1 = zeros([sz,2]);
        d1(:,:,1) = 2*M(:,:,2); d1(:,:,2) = Delta-M11mM22;
        x = norms(d1,2,3);
        d1(:,:,1) = d1(:,:,1)./x;d1(:,:,2) = d1(:,:,2)./x;
        x = x.^2; %= 2*Delta.*(Delta-M11mM22);
        ind = (Delta(:)<tol*max(Delta(:))) | (abs(x(:))<tol*max(x(:)));
        Delta(ind) = 0; x(ind) = 0;
        d1 = reshape(d1,[],2); d1(ind,2) = 0; d1(ind,1) = 1; d1 = reshape(d1,[sz,2]);
        d2 = zeros(size(d1)); d2(:,:,1) = d1(:,:,2); d2(:,:,2) = -d1(:,:,1);
        
        c1 = obj.C1(Delta,Sigma);c2 = obj.C2(Delta,Sigma);
        dc1 = obj.DC1(Delta,Sigma);dc2 = obj.DC2(Delta,Sigma);
        dc1(isnan(dc1)) = 0;dc2(isnan(dc2)) = 0;
        
        Dx = cat(3,-2*(Delta-M11mM22).^2./Delta, 16*M(:,:,2) - 8*M(:,:,2).*M11mM22./Delta, 2*(Delta-M11mM22).^2./Delta);
        Dx(isnan(Dx)) = 0;
        
        % \partial_i d1_j = Dd1(:,:,j,i)
        Dd1 = zeros([size(d1) 3]);
        Dd1(:,:,1,1) = -d1(:,:,1).*Dx(:,:,1)./(2*x);
        Dd1(:,:,2,1) = -d1(:,:,2).*Dx(:,:,1)./(2*x) + (M11mM22./Delta-1)./sqrt(x);
        Dd1(:,:,1,2) = -d1(:,:,1).*Dx(:,:,2)./(2*x) + 2./sqrt(x);
        Dd1(:,:,2,2) = -d1(:,:,2).*Dx(:,:,2)./(2*x) + 4*M(:,:,2)./(Delta.*sqrt(x));
        Dd1(:,:,1,3) = -d1(:,:,1).*Dx(:,:,3)./(2*x);
        Dd1(:,:,2,3) = -d1(:,:,2).*Dx(:,:,3)./(2*x) - (M11mM22./Delta-1)./sqrt(x);
        Dd2 = zeros(size(Dd1)); Dd2(:,:,1,:) = Dd1(:,:,2,:); Dd2(:,:,2,:) = -Dd1(:,:,1,:);
        

        J = zeros([sz,2,2]);
        % \partial_{kl}J_{ij} = DJ(:,:,k+l-1,i,j)
        DJ = zeros([sz,3,2,2]);

        for i=1:2
            for j=1:2
                J(:,:,i,j) = (c1.*d1(:,:,i).*d1(:,:,j) + c2.*d2(:,:,i).*d2(:,:,j));

                DJ(:,:,1,i,j) = (dc1(:,:,2) + M11mM22.*dc1(:,:,1)./Delta).*d1(:,:,i).*d1(:,:,j)...
                              + (dc2(:,:,2) + M11mM22.*dc2(:,:,1)./Delta).*d2(:,:,i).*d2(:,:,j)...
                              + c1.*(Dd1(:,:,i,1).*d1(:,:,j)+Dd1(:,:,j,1).*d1(:,:,i))...
                              + c2.*(Dd2(:,:,i,1).*d2(:,:,j)+Dd2(:,:,j,1).*d2(:,:,i));
                DJ(:,:,2,i,j) = 4*(M(:,:,2)./Delta).*dc1(:,:,1).*d1(:,:,i).*d1(:,:,j)...
                              + 4*(M(:,:,2)./Delta).*dc2(:,:,1).*d2(:,:,i).*d2(:,:,j)...
                              + c1.*(Dd1(:,:,i,2).*d1(:,:,j)+Dd1(:,:,j,2).*d1(:,:,i))...
                              + c2.*(Dd2(:,:,i,2).*d2(:,:,j)+Dd2(:,:,j,2).*d2(:,:,i));
                DJ(:,:,3,i,j) = (dc1(:,:,2) - M11mM22.*dc1(:,:,1)./Delta).*d1(:,:,i).*d1(:,:,j)...
                              + (dc2(:,:,2) - M11mM22.*dc2(:,:,1)./Delta).*d2(:,:,i).*d2(:,:,j)...
                              + c1.*(Dd1(:,:,i,3).*d1(:,:,j)+Dd1(:,:,j,3).*d1(:,:,i))...
                              + c2.*(Dd2(:,:,i,3).*d2(:,:,j)+Dd2(:,:,j,3).*d2(:,:,i));
            end
        end
        
        if nnz(ind)>0
            J = reshape(J,[],2,2);
            DJ = reshape(DJ,[],3,2,2);
            dc1 = reshape(dc1,[],2);
            J(ind,1,1) = c1(ind);J(ind,1,2) = 0;J(ind,2,1) = 0;J(ind,2,2) = c1(ind);
            DJ(ind,1,1,1) = dc1(ind,2);DJ(ind,2,1,1) = 0;DJ(ind,3,1,1) = dc1(ind,2);
            DJ(ind,1,2,1) = 0;DJ(ind,2,2,1) = 0;DJ(ind,3,2,1) = 0;
            DJ(ind,1,1,2) = 0;DJ(ind,2,1,2) = 0;DJ(ind,3,1,2) = 0;
            DJ(ind,1,2,2) = dc1(ind,2);DJ(ind,2,2,2) = 0;DJ(ind,3,2,2) = dc1(ind,2);

            J = reshape(J,[sz,2,2]);
            DJ = reshape(DJ,[sz,3,2,2]);
            dc1 = reshape(dc1,size(dc2));
        end
        
        DJM = @(V,D,adj) DHJvM(obj, DJ, Du_rho,V,D,adj);
        
        if nargout == 2
            J = @(u)cat(3,...
                J(:,:,1,1).*u(:,:,1)+J(:,:,1,2).*u(:,:,2),...
                J(:,:,2,1).*u(:,:,1)+J(:,:,2,2).*u(:,:,2));
        end
        
        
%     %% DEBUGGING
%         
%         for e=10.^(-3:-1:-10)
%         V = randn([size(u),1,1,2]);
%         JV = squeeze(sum(bsxfun(@times,J,reshape(V,[sz,1,2])),4));
%         H = @(x) norm(x(:))^2/2;
%         DH = reshape(JV,[sz,1,2,1]);
%         DJV = bsxfun(@times,DJ,V); DJV = squeeze(sum(DJV,5));
%         DHJV = bsxfun(@times,DJV,DH);DHJV = squeeze(sum(DHJV,4));
%         
%         DHJMV = DJM(V,DH,true);
%         
%         uP = u+e*randn(size(u));
%         DuP = obj.smooth_deriv(uP,'rho',1,1);
%         MP = cat(3, DuP(:,:,1).*DuP(:,:,1), DuP(:,:,1).*DuP(:,:,2),DuP(:,:,2).*DuP(:,:,2));
%         MP = obj.smooth_deriv(MP, 'sigma', 0);
%         
%         M11mM22P = MP(:,:,1)-MP(:,:,3);SigmaP = MP(:,:,1)+MP(:,:,3); DeltaP = sqrt(M11mM22P.^2+4*MP(:,:,2).^2);
%         d1P = zeros([sz,2]); d1P(:,:,1) = 2*MP(:,:,2); d1P(:,:,2) = DeltaP-M11mM22P;
%         xP = norms(d1P,2,3); d1P(:,:,1) = d1P(:,:,1)./xP;d1P(:,:,2) = d1P(:,:,2)./xP; xP = xP.^2;
%         ind = (DeltaP(:)<tol*max(DeltaP(:))) | (abs(xP(:))<tol*max(xP(:))); DeltaP(ind) = 0; xP(ind) = 0;
%         d1P = reshape(d1P,[],2); d1P(ind,2) = 0; d1P(ind,1) = 1; d1P = reshape(d1P,[sz,2]);
%         d2P = zeros(size(d1P)); d2P(:,:,1) = d1P(:,:,2); d2P(:,:,2) = -d1P(:,:,1);
%         c1P = obj.C1(DeltaP,SigmaP);c2P = obj.C2(DeltaP,SigmaP);
%         
%         JP = zeros([sz,2,2]);
%         for i=1:2;for j=1:2;JP(:,:,i,j) = (c1P.*d1P(:,:,i).*d1P(:,:,j) + c2P.*d2P(:,:,i).*d2P(:,:,j));end;end;
%         JP = reshape(JP,[],2,2);JP(ind,1,1)=c1P(ind);JP(ind,2,2)=c1P(ind);JP(ind,1,2)=0;JP(ind,2,1)=0;JP=reshape(JP,size(J));
% 
%         JVP = squeeze(sum(bsxfun(@times,JP,reshape(V,[sz,1,2])),4));
%         du = uP-u; dM = MP-M;dJ = JP-J;dJV = cat(3,dJ(:,:,1,1).*V(:,:,1)+dJ(:,:,1,2).*V(:,:,2),dJ(:,:,2,1).*V(:,:,1)+dJ(:,:,2,2).*V(:,:,2));dH=H(JVP)-H(JV);
% 
%         DHJMV = DJM(V,DH,true);
%         DJMVU = DJM(V,du,false);
% 
%         disp((DHJMV(:)'*du(:)-DJMVU(:)'*DH(:))/abs(DJMVU(:)'*DH(:)))
%         continue
%         
%         ind = [2,2];
% %         tmp = (dM - nabM(du))/norm(du(:));norm(tmp(:));
% %         tmp = (dJ(:,:,ind(1),ind(2)) - squeeze(sum(DJ(:,:,:,ind(1),ind(2)).*nabM(du),3)))/norm(du(:));
% %         tmp = (dJV(:,:,ind(1)) - squeeze(sum(DJV(:,:,:,ind(1)).*nabM(du),3)))/norm(du(:));
% %         tmp = (dH - sum(sum(squeeze(sum(sum(DH.*DJV,4).*nabM(du),3)))))/norm(du(:));
% %         tmp = (dH-sum(sum(sum(DHJV.*nabM(du)))))/norm(du(:));
% %         tmp = (dH-sum(DHJMV(:).*du(:)))/norm(du(:));
% %         tmp = (dH-sum(DH(:).*DJMVU(:)))/norm(du(:));
%         tmp(isnan(0*tmp)) = 0;
%         disp(log10([norm(dH(:))/norm(du(:)), norm(tmp(:))]))
% %         figure;
% % %         ax = [1,size(dJ,1),1,size(dJ,2),min(dJ(:))/norm(du(:)),max(dJ(:))/norm(du(:))];
% %         subplot(1,2,1);surf(dJ(:,:,ind(1),ind(2))/norm(du(:))); shading flat;%axis(ax);
% %         subplot(1,2,2);surf(DJM(:,:,ind(1),ind(2)).*du/norm(du(:))); shading flat;%axis(ax);
% %         surf(squeeze(dJ(:,:,ind(1),ind(2))-sum(DJ(:,:,:,ind(1),ind(2)).*nabM(du),3))/norm(du(:))); shading flat;
%         end
% %         pause(1);
        end
    end
end


function Du = grad(u,deriv, order)
    if deriv > 2
        error('Given order must be either 0, 1 or 2');
    end
    ndim = numel(size(u));
    switch deriv
        case 0
%             zero derivatives is identity
            Du = u;
        case 1
            switch order
                % First derivative with Neumann boundary and forward
                % differences
                case 1
                    switch ndim
                        case 1
                            Du = zeros(size(u,1),1);
                            Du(1:end-1) = u(2:end)-u(1:end-1);
                            Du(end) = 0;
                        case 2
                            % Du = [D1u, D2u]
                            Du = cat(3,u,u);
                            Du(1:end-1,:,1) = u(2:end,:)-u(1:end-1,:);
                            Du(end,:,1) = 0;
                            Du(:,1:end-1,2) = u(:,2:end)-u(:,1:end-1);
                            Du(:,end,2) = 0;
                        case 3
                            % Du = [D1u, D2u, D3u]
                            Du = zeros(size(u,1),size(u,3),size(u,3),3);
                            Du(1:end-1,:,:,1) = u(2:end,:,:)-u(1:end-1,:,:);
                            Du(end,:,:,1) = 0;
                            Du(:,1:end-1,:,2) = u(:,2:end,:)-u(:,1:end-1,:);
                            Du(:,end,:,2) = 0;
                            Du(:,:,1:end-1,3) = u(:,:,2:end)-u(:,:,1:end-1);
                            Du(:,:,end,3) = 0;
                    end
                % First derivative with Neumann boundary and centered
                % differences
                case 2
                    switch ndim
                        case 1
                            Du = zeros(size(u,1),1);
                            Du(2:end-1) = u(3:end)-u(1:end-2);
                            Du(1) = u(2)-u(1); Du(end) = u(end)-u(end-1);
                        case 2
                            % Du = [D1u, D2u] with periodic boundary
%                             Du = cat(3,...
%                                 u([2:end,1],:)-u([end,1:end-1],:),...
%                                 u(:,[2:end,1])-u(:,[end,1:end-1]));
                            Du = cat(3,...
                                u([2:end,end],:)-u([1,1:end-1],:),...
                                u(:,[2:end,end])-u(:,[1,1:end-1]));
                        case 3
                            % Du = [D1u, D2u, D3u]
                            Du = zeros(size(u,1),size(u,3),size(u,3),3);
                            Du(2:end-1,:,:,1) = u(3:end,:,:)-u(1:end-2,:,:);
                            Du(1,:,:,1) = u(2,:,:)-u(1,:,:);Du(end,:,:,1) = u(end,:,:)-u(end-1,:,:);
                            Du(:,2:end-1,:,2) = u(:,3:end,:)-u(:,1:end-2,:);
                            Du(:,1,:,2) = u(:,2,:)-u(:,1,:);Du(:,end,:,2) = u(:,end,:)-u(:,end-1,:);
                            Du(:,:,2:end-1,3) = u(:,:,3:end)-u(:,:,1:end-2);
                            Du(:,:,1,3) = u(:,:,2)-u(:,:,1);Du(:,:,end,3) = u(:,:,end)-u(:,:,end-1);
                    end
                    Du = Du/2;
            end
        case 2
            % Second derivative with Neumann boundary and centered
            % differences
            switch ndim
                case 1
                    Du = zeros(size(u,1),1);
                    Du(2:end-1) = u(3:end) - 2*u(2:end-1) - u(1:end-2);
                    Du(1) = u(2)-u(1);
                    Du(end) = u(end-1)-u(end);
                case 2
                    Du = zeros(size(u,1),size(u,2),3);
                    % Du = [D11, D12, D22]
                    Du(2:end-1,:,1) = u(3:end,:) - 2*u(2:end-1,:) - u(1:end-2,:);
                    Du(1,:,1) = u(2,:)-u(1,:);
                    Du(end,:,1) = u(end-1,:)-u(end,:);
                    
                    Du(2:end-1,2:end-1,2) = 0.25*(u(3:end,3:end)-u(3:end,1:end-2)-u(1:end-2,3:end)+u(1:end-2,1:end-2));
                    Du(2:end-1,1,2) = 0.25*(u(3:end,2)-u(3:end,1)-u(1:end-2,2)+u(1:end-2,1));
                    Du(2:end-1,end,2) = 0.25*(u(3:end,end)-u(3:end,end-1)-u(1:end-2,end)+u(1:end-2,end-1));
                    Du(1,2:end-1,2) = 0.25*(u(2,3:end)-u(2,1:end-2)-u(1,3:end)+u(1,1:end-2));
                    Du(end,2:end-1,2) = 0.25*(u(end,3:end)-u(end,1:end-2)-u(end-1,3:end)+u(end-1,1:end-2));
                    Du(1,1,2) = 0.25*(u(2,2)-u(2,1)-u(1,2)+u(1,1));
                    Du(end,1,2) = 0.25*(u(end,2)-u(end,1)-u(end-1,2)+u(end-1,1));
                    Du(1,end,2) = 0.25*(u(2,end)-u(2,end-1)-u(1,end)+u(1,end-1));
                    Du(end,end,2) = 0.25*(u(end,end)-u(end,end-1)-u(end-1,end)+u(end-1,end-1));
                    
                    Du(:,2:end-1,3) = u(:,3:end) - 2*u(:,2:end-1) - u(:,1:end-2);
                    Du(:,1,3) = u(:,2)-u(:,1);
                    Du(:,end,3) = u(:,end-1)-u(:,end);
                case 3
                    Du = zeros(size(u,1),size(u,3),size(u,3),6);
                    % Du = [D11, D12, D13, D22, D23, D33]
                    Du(2:end-1,:,:,1) = u(3:end,:,:) - 2*u(2:end-1,:,:) - u(1:end-2,:,:);
                    Du(1,:,:,1) = u(2,:,:)-u(1,:,:);
                    Du(end,:,:,1) = u(end-1,:,:)-u(end,:,:);
                    
                    Du(2:end-1,2:end-1,:,2) = 0.25*(u(3:end,3:end,:)-u(3:end,1:end-2,:)-u(1:end-2,3:end,:)+u(1:end-2,1:end-2,:));
                    Du(2:end-1,1,:,2) = 0.25*(u(3:end,2,:)-u(3:end,1,:)-u(1:end-2,2,:)+u(1:end-2,1,:));
                    Du(2:end-1,end,:,2) = 0.25*(u(3:end,end)-u(3:end,end-1)-u(1:end-2,end)+u(1:end-2,end-1,:));
                    Du(1,2:end-1,:,2) = 0.25*(u(2,3:end,:)-u(2,1:end-2,:)-u(1,3:end,:)+u(1,1:end-2,:));
                    Du(end,2:end-1,:,2) = 0.25*(u(end,3:end)-u(end,1:end-2)-u(end-1,3:end)+u(end-1,1:end-2));
                    Du(1,1,:,2) = 0.25*(u(2,2,:)-u(2,1,:)-u(1,2,:)+u(1,1,:));
                    Du(end,1,:,2) = 0.25*(u(end,2,:)-u(end,1,:)-u(end-1,2,:)+u(end-1,1,:));
                    Du(1,end,:,2) = 0.25*(u(2,end,:)-u(2,end-1,:)-u(1,end,:)+u(1,end-1,:));
                    Du(end,end,:,2) = 0.25*(u(end,end,:)-u(end,end-1,:)-u(end-1,end,:)+u(end-1,end-1,:));

                    Du(2:end-1,:,2:end-1,3) = 0.25*(u(3:end,:,3:end)-u(3:end,:,1:end-2)-u(1:end-2,:,3:end)+u(1:end-2,:,1:end-2));
                    Du(2:end-1,:,1,3) = 0.25*(u(3:end,:,2)-u(3:end,:,1)-u(1:end-2,:,2)+u(1:end-2,:,1));
                    Du(2:end-1,:,end,3) = 0.25*(u(3:end,:,end)-u(3:end,:,end-1)-u(1:end-2,:,end)+u(1:end-2,:,end-1));
                    Du(1,:,2:end-1,3) = 0.25*(u(2,:,3:end)-u(2,:,1:end-2)-u(1,:,3:end)+u(1,:,1:end-2));
                    Du(end,:,2:end-1,3) = 0.25*(u(end,:,3:end)-u(end,:,1:end-2)-u(end-1,:,3:end)+u(end-1,:,1:end-2));
                    Du(1,:,1,3) = 0.25*(u(2,:,2)-u(2,:,1)-u(1,:,2)+u(1,:,1));
                    Du(end,:,1,3) = 0.25*(u(end,:,2)-u(end,:,1)-u(end-1,:,2)+u(end-1,:,1));
                    Du(1,:,end,3) = 0.25*(u(2,:,end)-u(2,:,end-1)-u(1,:,end)+u(1,:,end-1));
                    Du(end,:,end,3) = 0.25*(u(end,:,end)-u(end,:,end-1)-u(end-1,:,end)+u(end-1,:,end-1));

                    Du(:,2:end-1,:,4) = u(:,3:end,:) - 2*u(:,2:end-1,:) - u(:,1:end-2,:);
                    Du(:,1,:,4) = u(:,2,:)-u(:,1,:);
                    Du(:,end,:,4) = u(:,end-1,:)-u(:,end,:);
                    
                    Du(:,2:end-1,2:end-1,5) = 0.25*(u(:,3:end,3:end)-u(:,3:end,1:end-2)-u(:,1:end-2,3:end)+u(:,1:end-2,1:end-2));
                    Du(:,2:end-1,1,5) = 0.25*(u(:,3:end,2)-u(:,3:end,1)-u(:,1:end-2,2)+u(:,1:end-2,1));
                    Du(:,2:end-1,end,5) = 0.25*(u(:,3:end,end)-u(:,3:end,end-1)-u(:,1:end-2,end)+u(:,1:end-2,end-1));
                    Du(:,1,2:end-1,5) = 0.25*(u(:,2,3:end)-u(:,2,1:end-2)-u(:,1,3:end)+u(:,1,1:end-2));
                    Du(:,end,2:end-1,5) = 0.25*(u(:,end,3:end)-u(:,end,1:end-2)-u(:,end-1,3:end)+u(:,end-1,1:end-2));
                    Du(:,1,1,5) = 0.25*(u(:,2,2)-u(:,2,1)-u(:,1,2)+u(:,1,1));
                    Du(:,end,1,5) = 0.25*(u(:,end,2)-u(:,end,1)-u(:,end-1,2)+u(:,end-1,1));
                    Du(:,1,end,5) = 0.25*(u(:,2,end)-u(:,2,end-1)-u(:,1,end)+u(:,1,end-1));
                    Du(:,end,end,5) = 0.25*(u(:,end,end)-u(:,end,end-1)-u(:,end-1,end)+u(:,end-1,end-1));
                    
                    Du(:,:,2:end-1,6) = u(:,:,3:end) - 2*u(:,:,2:end-1) - u(:,:,1:end-2);
                    Du(:,:,1,6) = u(:,:,2)-u(:,:,1);
                    Du(:,:,end,6) = u(:,:,end-1)-u(:,:,end);
            end
    end    
end

function v = DM(obj, Du_rho, d,adj)
% M(u) = (\partial_iu_\rho \partial_ju_\rho)_\sigma
% DM(u)d = (\partial_iu_\rho \partial_jd_\rho +
%   \partial_ju_\rho \partial_id_\rho)_\sigma
% DM   = conv(sigma)[diag(\partial_iu_rho)D_j + diag(\partial_ju_rho)D_i]conv(rho)
% DM^T = conv(rho)[D_j^Tdiag(\partial_iu_rho) + D_i^Tdiag(\partial_ju_rho)]conv(sigma)

    if adj
        d = obj.smooth_deriv(d,'sigma',0);
        
        tmp = 2*Du_rho(:,:,1).*d(:,:,1);
        tmp(end,:) = tmp(end-1,:);
        tmp(2:end-1,:) = tmp(1:end-2,:)-tmp(2:end-1,:);
        tmp(1,:) = -tmp(1,:);
        v = tmp;

        tmp = Du_rho(:,:,2).*d(:,:,2);
        tmp(end,:) = tmp(end-1,:);
        tmp(2:end-1,:) = tmp(1:end-2,:)-tmp(2:end-1,:);
        tmp(1,:) = -tmp(1,:);
        v = v + tmp;
        tmp = Du_rho(:,:,1).*d(:,:,2);
        tmp(:,end) = tmp(:,end-1);
        tmp(:,2:end-1) = tmp(:,1:end-2)-tmp(:,2:end-1);
        tmp(:,1) = -tmp(:,1);
        v = v + tmp;
        
        tmp = 2*Du_rho(:,:,2).*d(:,:,3);
        tmp(:,end) = tmp(:,end-1);
        tmp(:,2:end-1) = tmp(:,1:end-2)-tmp(:,2:end-1);
        tmp(:,1) = -tmp(:,1);
        v = v + tmp;

        v = obj.smooth_deriv(v,'rho',0);
    else
        d = obj.smooth_deriv(d,'rho',1,1);

        v = cat(3, 2*Du_rho(:,:,1).*d(:,:,1),...
                   Du_rho(:,:,1).*d(:,:,2)+Du_rho(:,:,2).*d(:,:,1),...
                   2*Du_rho(:,:,2).*d(:,:,2));
               
        v = obj.smooth_deriv(v,'sigma',0);
    end
end

function d = DHJvM(obj,DJ, Du_rho,v,D,adj)
% D_{kl}J_{ij} = DJ(:,:,k+l-1,i,j)
% D(H(J(M(u))v))du = sum DH_i D_{kl}J_{ij}v_j D_mM_{kl}du_m
% nabla(H(J(M(u))v))\cdot du = sum \nabla_mM_{kl} \nabla_{kl}J_{ij}v_j \nabla H_i du_m
    
    v = reshape(v, size(DJ,1),size(DJ,2),1,1,size(DJ,5));
    DJV = squeeze(sum(bsxfun(@times,DJ,v),5));

    if adj
        % d = @(nabla H) nabla(H(J(M(u))v))
%           = DM^T DJV^T nabla H
        DH = reshape(D,size(DJ,1),size(DJ,2),1,size(DJ,4),1);
        DJVH = bsxfun(@times,DJV,DH);
        DJVH = squeeze(sum(DJVH,4));
        
        d = DM(obj,Du_rho,DJVH,true);

%         %d_{kl,i} = d(:,:,k+l-1,i)
%         d = zeros([size(DJVH,1),size(DJVH,2),3,size(DJVH,4)]);
% 
%         for k=1:size(DJVH,4)
%         tmp = Du_rho(:,:,1).*obj.smooth_deriv(DJVH(:,:,1,k),'sigma',0);
%         Dtmp = zeros(size(tmp));
%         Dtmp(2:end-1,:) = tmp(1:end-2,:)-tmp(2:end-1,:);
%         Dtmp(1,:)=-tmp(1,:); Dtmp(end,:) = tmp(end,:);
%         d(:,:,1,k) = 2*obj.smooth_deriv(Dtmp,'rho',0);
% 
%         tmp = Du_rho(:,:,2).*obj.smooth_deriv(DJVH(:,:,2,k),'sigma',0);
%         Dtmp = zeros(size(tmp));
%         Dtmp(2:end-1,:) = tmp(1:end-2,:)-tmp(2:end-1,:);
%         Dtmp(1,:)=-tmp(1,:); Dtmp(end,:) = tmp(end,:);
%         tmp1 = Dtmp;tmp = Du_rho(:,:,1).*obj.smooth_deriv(DJVH(:,:,2,k),'sigma',0);
%         Dtmp(:,2:end-1) = tmp(:,1:end-2)-tmp(:,2:end-1);
%         Dtmp(:,1)=-tmp(:,1); Dtmp(:,end) = tmp(:,end);
%         d(:,:,2,k) = obj.smooth_deriv(tmp1+Dtmp,'rho',0);
% 
%         tmp = Du_rho(:,:,2).*obj.smooth_deriv(DJVH(:,:,3,k),'sigma',0);
%         Dtmp = zeros(size(tmp));
%         Dtmp(:,2:end-1) = tmp(:,1:end-2)-tmp(:,2:end-1);
%         Dtmp(:,1)=-tmp(:,1); Dtmp(:,end) = tmp(:,end);
%         d(:,:,3,k) = 2*obj.smooth_deriv(Dtmp,'rho',0);
%         end
% 
%         d = squeeze(sum(d,3));
        d(isnan(d)) = 0;
    else
        % d = @(du) D(J(M(u))v)du
        du = reshape(D,size(DJ,1),size(DJ,2));
        d = squeeze(sum(bsxfun(@times,DJV,...
            reshape(DM(obj,Du_rho,du,false),size(DJV,1),size(DJV,2),3,1)...
            ),3));
    end
end