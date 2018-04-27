classdef solver < handle
%   Solver to compute 
%   $$ \min_{u\geq0} \frac12\|SRu-f|^2 + \alpha\|D\nabla Ru\|_{2,1} + \beta\|\nabla u|_{2,1}$$
    properties(GetAccess = public, SetAccess = protected)
        f          % projection data
        diffusion  % diffusion_map object to use
        in_sz      % size of recorded data
        recon_sz   % size of reconstruction
        sino_sz    % size of full sinogram
        scale      % normalisation of data
        K          % matrix for primal formulation
        subR       % matrix for subsampled Radon transform
        fullR      % matrix for fully sampled Radon transform
        angle_mask % Set of indices s.t. subR = fullR(angle_mask,:)
        gradu      % matrix for gradient of image
        gradRu     % matrix for gradient of sinogram
        diffop     % matrix for D\nabla
        normK      % estimation of 2-norm of K
        solverOpts % optional parameters for solver
        stepRatio  % square of the ratio between steps
        var        % store of primal/dual iterates
        prevvar    % store of previous primal/dual iterates
    end

    methods(Static)
        function p = getParser(justStrings)
            if nargin == 0
                justStrings = false;
            end
            
            if justStrings
                p = {'scale', 'recon_sz','stepRatio','maxiter', 'gamma', 'delta', 'sinoW', 'imW'};
            else
                p = inputParser;p.KeepUnmatched = true;
                p.addParameter('scale','default')
                p.addParameter('recon_sz','default')
                p.addParameter('stepRatio',1)
                p.addParameter('maxiter',1000)
                p.addParameter('gamma',0.95);
                p.addParameter('delta',0.95);
                p.addParameter('sinoW',1);
                p.addParameter('imW',1);
            end
        end
    end
 
    methods(Access=protected)
        %% Personal child functions
        function makeMatrices(obj,subangles,fullangles)
            % Angles must be in radons!
            subangles = subangles*pi/180;
            fullangles = fullangles*pi/180;
            stretch_sz = numel(radon(zeros(obj.recon_sz),45));
            obj.sino_sz = [numel(fullangles),stretch_sz];
            obj.angle_mask = zeros(size(subangles));
            for i=1:numel(subangles);[~,obj.angle_mask(i)] = min(abs(fullangles-subangles(i)));end;
            
            % Initiate fully sampled Radon operator
            proj_geom = astra_create_proj_geom('parallel', 1, stretch_sz, fullangles);
            vol_geom  = astra_create_vol_geom(obj.recon_sz(1),obj.recon_sz(2));
            Rop = opTomo('cuda', proj_geom, vol_geom);
            LHS = 1./sqrt(Rop*ones(size(Rop,2),1)); RHS = 1./sqrt(Rop'*ones(size(Rop,1),1));
            % Normalise full Radon transform
            S = double(min(LHS)*min(RHS)); RopConj = Rop';
            obj.fullR = fcthdlop([size(Rop, 2) 1], [size(Rop,1) 1],...
                @(z)(Rop*z)*S, @(z)RopConj*(z*S));
%             obj.fullR = load('radonmatrix_size64_angles_180.mat','R');
%             obj.fullR = obj.fullR.R;
%             obj.fullR = obj.toMatrix(@(x) obj.fullR*x(:), [size(obj.fullR,2),1]);

            % Initiate sub-sampled Radon operator
            proj_geom = astra_create_proj_geom('parallel', 1, stretch_sz, subangles);
            vol_geom  = astra_create_vol_geom(obj.recon_sz(1),obj.recon_sz(2));
            Rop = opTomo('cuda', proj_geom, vol_geom);
%             LHS = 1./sqrt(Rop*ones(size(Rop,2),1)); RHS = 1./sqrt(Rop'*ones(size(Rop,1),1));
            % Normalise sub Radon transform
%             S = double(min(LHS)*min(RHS)); 
            RopConj = Rop';
            obj.subR = fcthdlop([size(Rop, 2) 1], [size(Rop,1) 1],...
                @(z)(Rop*z)*S, @(z)RopConj*(z*S));
%             obj.subR = obj.fullR(obj.angle_mask,:);
            
            % Initiate gradient operators
%             which('Grad')
            obj.gradu = Grad(obj.recon_sz,false);
            obj.gradRu = Grad(obj.sino_sz,false);
            
            if numel(obj.diffusion)>1
                % Initiate difussion field operator
                D = fcthdlop([size(obj.gradRu,1),1],[size(obj.gradRu,1),1],...
                    @obj.fwrd_diffusion, @obj.bwrd_diffusion);
                obj.diffop = D*obj.gradRu;

                obj.K = [obj.subR;obj.gradu;obj.diffop*obj.fullR];
                obj.normK = sqrt(1+1+max(obj.diffusion.c1_val(:))^2);
            else
                obj.K = [obj.subR;obj.gradu];
                obj.normK = sqrt(1+1);
            end
        end
        function v = fwrd_diffusion(obj,Du)
            sz = size(obj.diffusion.c1_val);
            Du = reshape(Du,sz(1),sz(2),2);
            v = cat(3,...
                obj.diffusion.c1_val.*(obj.diffusion.e1_val(:,:,1).*Du(:,:,1) + obj.diffusion.e1_val(:,:,2).*Du(:,:,2)),...
                obj.diffusion.c2_val.*(obj.diffusion.e2_val(:,:,1).*Du(:,:,1) + obj.diffusion.e2_val(:,:,2).*Du(:,:,2)));
%             v = cat(3,...
%                 obj.diffusion.c1_val.*(obj.diffusion.e1_val(:,:,1).*obj.diffusion.e1_val(:,:,1).*Du(:,:,1) + obj.diffusion.e1_val(:,:,1).*obj.diffusion.e1_val(:,:,2).*Du(:,:,2)) + obj.diffusion.c2_val.*(obj.diffusion.e2_val(:,:,1).*obj.diffusion.e2_val(:,:,1).*Du(:,:,1) + obj.diffusion.e2_val(:,:,1).*obj.diffusion.e2_val(:,:,2).*Du(:,:,2)),...
%                 obj.diffusion.c1_val.*(obj.diffusion.e1_val(:,:,2).*obj.diffusion.e1_val(:,:,1).*Du(:,:,1) + obj.diffusion.e1_val(:,:,2).*obj.diffusion.e1_val(:,:,2).*Du(:,:,2)) + obj.diffusion.c2_val.*(obj.diffusion.e2_val(:,:,2).*obj.diffusion.e2_val(:,:,1).*Du(:,:,1) + obj.diffusion.e2_val(:,:,2).*obj.diffusion.e2_val(:,:,2).*Du(:,:,2)));
            v = v(:);
        end
        function v = bwrd_diffusion(obj,u)
            sz = size(obj.diffusion.c1_val);
            u = reshape(u,sz(1),sz(2),2);
            u(:,:,1) = obj.diffusion.c1_val.*u(:,:,1);
            u(:,:,2) = obj.diffusion.c2_val.*u(:,:,2);
            v = cat(3,...
                obj.diffusion.e1_val(:,:,1).*u(:,:,1) + obj.diffusion.e2_val(:,:,1).*u(:,:,2),...
                obj.diffusion.e1_val(:,:,2).*u(:,:,1) + obj.diffusion.e2_val(:,:,2).*u(:,:,2));
%             v = cat(3,...
%                 obj.diffusion.c1_val.*(obj.diffusion.e1_val(:,:,1).*obj.diffusion.e1_val(:,:,1).*u(:,:,1) + obj.diffusion.e1_val(:,:,1).*obj.diffusion.e1_val(:,:,2).*u(:,:,2)) + obj.diffusion.c2_val.*(obj.diffusion.e2_val(:,:,1).*obj.diffusion.e2_val(:,:,1).*u(:,:,1) + obj.diffusion.e2_val(:,:,1).*obj.diffusion.e2_val(:,:,2).*u(:,:,2)),...
%                 obj.diffusion.c1_val.*(obj.diffusion.e1_val(:,:,2).*obj.diffusion.e1_val(:,:,1).*u(:,:,1) + obj.diffusion.e1_val(:,:,2).*obj.diffusion.e1_val(:,:,2).*u(:,:,2)) + obj.diffusion.c2_val.*(obj.diffusion.e2_val(:,:,2).*obj.diffusion.e2_val(:,:,1).*u(:,:,1) + obj.diffusion.e2_val(:,:,2).*obj.diffusion.e2_val(:,:,2).*u(:,:,2)));
            v = v(:);
        end
    end
    methods(Access=public)
        %% Constructor:
        function obj = solver(f, subangles, fullangles, diffusion, varargin)
            p = solver.getParser();
            p.parse(varargin{:})
            p = p.Results;
            
            in_sz = size(f);
            recon_sz = p.recon_sz;
            if strcmp(recon_sz,'default')
                recon_sz = in_sz;
            end
            
            obj.diffusion = diffusion;
            obj.scale = p.scale;
            obj.stepRatio = p.stepRatio;
            obj.solverOpts = p;
            obj.in_sz = in_sz;
            obj.recon_sz = recon_sz;
            %%             Start the model
            obj.makeMatrices(subangles, fullangles);
            obj.setData(f);
        end
        
        %% Setters and getters:
        function solve(obj,alpha,beta, quiet)
            if nargin == 3
                quiet = false;
            end
            if alpha == 0
                % Just TV
                if beta == 0
                    CTV_PDHG(obj,false, quiet);
                else
                    TV_PDHG(obj,sqrt(beta),false, quiet);
                end
            elseif beta == 0
                % Just sinogram diffusion
                TV_PDHG(obj,sqrt(alpha),true, quiet);
            elseif numel([alpha(:);beta(:)]) == 2
                % Combined problem
                TV_DTV_PDHG(obj, [1,alpha,beta]/sqrt(min(alpha,beta)), quiet);
            elseif numel(alpha) == 2 && numel(beta) == 2
                % Full problem
                sub = [alpha(:);beta(:)]; sub = sub(sub<inf);
                Symmetric_PDHG(obj, [alpha(:);beta(:)]/(prod(sub).^(1/numel(sub))), quiet);
            end
        end
        function [sigma,tau] = getStepSizes(obj, normK)
            if nargin == 1; normK = obj.normK;end
            normAmp = 2;
            tau = normAmp*obj.stepRatio/normK;
            sigma = normAmp/(normK*obj.stepRatio);
        end
        function x = getResult(obj)
            x = reshape(obj.var.x(1:prod(obj.recon_sz)),obj.recon_sz)/obj.scale;
        end
        function sz = size(obj, arg)
            if nargin == 1
                sz = [prod(obj.in_sz), prod(obj.recon_sz)];
            elseif nargin > 2
                error('Too many input arguments')
            elseif arg == 1
                sz = prod(obj.in_sz);
            elseif arg == 2
                sz = prod(obj.recon_sz);
            else
                error('Index is not valid')
            end
        end
        function v = getFidelity(obj, p)
            v = obj.subR*obj.var.x-obj.f;
            if nargin == 2
                if p == 2
                    v = norm(v(:),2)^2/2;
                else
                    v = norm(v(:),p);
                end
                v = v/numel(obj.f);
            end
        end
        function setData(obj, f)
            % Rescale data
            if ~isnumeric(obj.scale);obj.scale = 2/max(f(:));end
            obj.scale = 1;
            obj.f = double(f(:)*obj.scale);
        end
        function warmStart(obj, x)
            obj.var.x = x(:)*obj.scale;
            if ~any(strcmp('y',fieldnames(obj.var)))
                obj.var.y(:) = 0;
            end
            obj.prevvar = obj.var;
        end
        function R = toMatrix(~,f,sz)
            e = zeros(sz);
            R = zeros(numel(f(e)), prod(sz));
            tic
            for ii = 1:prod(sz)
                e(ii) = 1;
                R(:,ii) = reshape(f(e),[],1);
                e(ii) = 0;
            end
            R = sparse(R);
            disp(['Matrix calculation complete after ', num2str(toc), 's']);
        end
    end
    
end

function CTV_PDHG(obj,structured,quiet)
%% Constrained TV reconstruction
% % Minimise \|Du\|_{2,1} s.t. Ru = f
% % We split this into:
% % $F(x) = \chi_{x(1)=f} + \|x(2)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0}$
% % $F^*(y) = <y(1),f> + \chi_{|y(2)|_2\leq\lambda}$
% % $G^*(v) = \chi_{v\leq 0}$
% % $prox_{tF^*}(Y) = [Y(1)-tf, Y(2)/max(1,|Y(2)|_2)]$
% % $prox_{sG}(X) = max(0,X)$


    if structured
        K = [obj.subR;obj.diffop*obj.fullR]; Kconj = K';
        D = obj.diffop;
        normK = sqrt(1+max(obj.diffusion.c1_val(:))^2);
    else
        K = [obj.subR;obj.gradu]; Kconj = K';
        D = obj.gradu;
        normK = sqrt(2);
    end
    
    cut = prod(obj.in_sz);
    if isempty(obj.var)
        obj.var.x = zeros([size(K, 2) 1]);
        obj.var.y = zeros([size(K, 1) 1]);
    else
        if any(size(obj.var.x) - [size(K, 2), 1])
            obj.var.x = zeros([size(K, 2), 1]);
        end
        if any(size(obj.var.y) - [size(K, 1), 1])
            obj.var.y = zeros([size(K, 1), 1]);
        end
    end
    if isempty(obj.prevvar)
        obj.prevvar = obj.var;
    end
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-8,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = obj.solverOpts.sinoW(:);iW = obj.solverOpts.imW(:);
    
    if ~quiet;fig = figure;end;k=0;inner_loop = 500;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        
        % Primal iteration: just a threshold
        obj.var.x = max(0, obj.var.x-s*prevKStary);
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut)+obj.subR*xbar;
        y2 = reshape(obj.var.y(cut+1:end)+D*xbar,[],2);
        
        y1 = y1-t*obj.f;
        y2 = normProj(y2,iW/1000);
        obj.var.y = [y1(:);y2(:)];
        
        KStary = Kconj*obj.var.y(:);
        
       
        % Adaptive step size:
        d = (2/gamma)*s*t*(obj.var.x-obj.prevvar.x)'*(KStary-prevKStary)/...
            (t*norm(obj.var.x-obj.prevvar.x)^2 + s*norm(obj.var.y-obj.prevvar.y)^2);
        if d > 1
            s = s*delta/d;
            t = t*delta/d;
        else
            s = s*1.001;
            t = t*1.001;
        end


        k = k+1;
        prevKStary = KStary;
    end
        if ~quiet
        fig;
        subplot(2,2,1);imagesc(reshape(obj.var.x,obj.recon_sz));
        title(['Iteration ' num2str(k)]);
        subplot(2,2,2);imagesc(reshape(obj.fullR*obj.var.x(:),[],obj.in_sz(2))');
        subplot(2,2,3);imagesc(abs(reshape(obj.subR*obj.var.x(:)-obj.f,[],obj.in_sz(2))'));title('Sino error');colorbar;
        subplot(2,2,4);imagesc(reshape(obj.f,[],obj.in_sz(2))');title('Given sino chunk')
        drawnow
        end


        % Primal-Dual Gap calculation:
        p = [sum(iW.*norms(reshape(D*obj.var.x,[],2),2,2)),max(abs(obj.subR*obj.var.x-obj.f))];
        d = [sum(y1.*obj.f) , max(0,-min(KStary))];
        pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
        end
        pd_gap = max(abs(pd_gap));

        % Sensitivity calculation:
        normX = norm(obj.var.x,inf);normY = norm(obj.var.y,inf);
        if normX > eps && normY > eps
            sensitivity = norm(obj.var.x - obj.prevvar.x,inf)/normX ...
                + norm(obj.var.y - obj.prevvar.y,inf)/normY;
        else
            sensitivity = Inf;
        end
    
    end


    if sensitivity <= tol(1)
        disp(['Change between iterations dropped below sensitivity barrier after ', num2str(k), ' iterations'])
    end
    if pd_gap <= tol(2)
        disp(['Primal dual gap converged after ', num2str(k), ' iterations'])
    else
        disp(['Final primal-dual gap: ', num2str(pd_gap)])
    end
    
    if ~quiet && isvalid(fig)
        close(fig);
    end

end

function TV_PDHG(obj, lambda, structured, quiet)
%% TV reconstruction
% % Minimise \frac{1}{2\lambda}\|Ru-f\|^2 + \lambda\|Du\|_{2,1}
% % We split this into:
% % $F(x) = \frac{1}{2\lambda}\|x(1)-f\|^2 + \lambda\|x(2)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0}$
% % $F^*(y) = \frac{\lambda}{2}\|y(1)\|^2 + <y(1),f> + \chi_{|y(2)|_2\leq\lambda}$
% % $G^*(v) = \chi_{v\leq 0}$
% % $prox_{tF^*}(Y) = [(Y(1)-tf)/(1+t\lambda), Y(2)/max(1,|Y(2)|_2/\lambda)]$
% % $prox_{sG}(X) = max(0,X)$

    if structured
        K = [obj.subR;obj.diffop*obj.fullR]; Kconj = K';
        D = obj.diffop;
        normK = sqrt(1+max(obj.diffusion.c1_val(:))^2);
    else
        K = [obj.subR;obj.gradu]; Kconj = K';
        D = obj.gradu;
        normK = sqrt(2);
    end
    
    cut = prod(obj.in_sz);
    if isempty(obj.var)
        obj.var.x = zeros([size(K, 2) 1]);
        obj.var.y = zeros([size(K, 1) 1]);
    else
        if any(size(obj.var.x) - [size(K, 2), 1])
            obj.var.x = zeros([size(K, 2), 1]);
        end
        if any(size(obj.var.y) - [size(K, 1), 1])
            obj.var.y = zeros([size(K, 1), 1]);
        end
    end
    if isempty(obj.prevvar)
        obj.prevvar = obj.var;
    end
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-8,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = obj.solverOpts.sinoW(:);iW = obj.solverOpts.imW(:);
    
    if ~quiet;fig = figure;end;k=0;inner_loop = 100;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        
        % Primal iteration: just a threshold
        obj.var.x = max(0, obj.var.x-s*prevKStary);
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut)+obj.subR*xbar;
        y2 = reshape(obj.var.y(cut+1:end)+D*xbar,[],2);
        
        y1 = (y1-t*obj.f)./(1+(t*lambda)./sW);
        y2 = normProj(y2,lambda*iW);
        obj.var.y = [y1(:);y2(:)];
        
        KStary = Kconj*obj.var.y(:);
        
       
        % Adaptive step size:
        d = (2/gamma)*s*t*(obj.var.x-obj.prevvar.x)'*(KStary-prevKStary)/...
            (t*norm(obj.var.x-obj.prevvar.x)^2 + s*norm(obj.var.y-obj.prevvar.y)^2);
        if d > 1
            s = s*delta/d;
            t = t*delta/d;
        end


        k = k+1;
        prevKStary = KStary;
    end
        if ~quiet
        fig;
        subplot(2,2,1);imagesc(reshape(obj.var.x,obj.recon_sz));
        title(['Iteration ' num2str(k)]);
        subplot(2,2,2);imagesc(reshape(obj.fullR*obj.var.x(:),[],obj.in_sz(2))');title('full sino')
        subplot(2,2,3);imagesc(abs(reshape(obj.subR*obj.var.x(:)-obj.f,[],obj.in_sz(2)))');colorbar;title('|SRx-f|');
        subplot(2,2,4);imagesc(reshape(obj.f,[],obj.in_sz(2))');title('f');
        drawnow
        end


        % Primal-Dual Gap calculation:
        p = [sum(sW.*(obj.subR*obj.var.x-obj.f).^2)/(2*lambda) + lambda*sum(iW.*norms(reshape(D*obj.var.x,[],2),2,2)),0];
        d = [sum(y1.*((lambda/2)*(y1./sW) + obj.f)) , max(0,-min(KStary))];
        pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
%         pd_gap = [sum((obj.subR*obj.var.x-obj.f).^2/(2*lambda) + y1.*((lambda/2)*y1 + obj.f)) + lambda*sum(norms(reshape(obj.gradu*obj.var.x,[],2),2,2)),max(0,-min(KStary)-10*sqrt(eps))];
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
%             ip_gap = sum(obj.var.x.*KStary)-sum(obj.var.y.*(K*obj.var.x));
%             disp(['ip_gap: ', num2str(abs(ip_gap)/(abs(p(1))+abs(p(2))))]);
%             fprintf('%f %f %f %f\n',sum((obj.subR*obj.var.x(:)-obj.f(:)).^2)/2,lambda^2*sum(norms(reshape(D*obj.var.x(:),[],2),2,2)),(1/2)*sum(y1(:).^2) + sum(y1(:).*obj.f(:)),-min(KStary))
%             fprintf('%f %f %f %f\n', p(1), d(1), p(1)+d(1), pd_gap(1));
        end
        pd_gap = max(abs(pd_gap));

        % Sensitivity calculation:
        normX = norm(obj.var.x,inf);normY = norm(obj.var.y,inf);
        if normX > eps && normY > eps
            sensitivity = norm(obj.var.x - obj.prevvar.x,inf)/normX ...
                + norm(obj.var.y - obj.prevvar.y,inf)/normY;
        else
            sensitivity = Inf;
        end
    
    end

    if sensitivity <= tol(1)
        disp(['Change between iterations dropped below sensitivity barrier after ', num2str(k), ' iterations'])
    end
    if pd_gap <= tol(2)
        disp(['Primal dual gap converged after ', num2str(k), ' iterations'])
    else
        disp(['Final primal-dual gap: ', num2str(pd_gap)])
    end
    
    if ~quiet && isvalid(fig)
        close(fig);
    end

end

function TV_DTV_PDHG(obj, lambda, quiet)
%% TV + DTV reconstruction
% % Minimise \frac{\lambda_1}{2}\|Ru-f\|^2 + \lambda_2\|Du\|_{2,1} + \lambda_3\|Ju\|_{2,1}
% % We split this into:
% % $F(x) = \frac{\lambda_1}{2}\|x(1)-f\|^2 + \lambda_2\|x(2)\|_{2,1}+ \lambda_3\|x(3)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0}$
% % $F^*(y) = \frac{1}{2\lambda_1}\|y(1)\|^2 + <y(1),f> + \chi_{|y(i)|_2\leq\lambda_i}$
% % $G^*(v) = \chi_{v\leq 0}$
% % $prox_{tF^*}(Y) = [(Y(1)-tf)/(1+t/\lambda_1), Y(i)/max(1,|Y(i)|_2/\lambda_i)]$
% % $prox_{sG}(X) = max(0,X)$

    s = normest(obj.diffop)/300;
    K = [obj.subR;obj.gradu;(1/s)*obj.diffop*obj.fullR]; Kconj = K';
    D = @(x) (1/s)*(obj.diffop*(obj.fullR*x));
    lambda(3) = lambda(3)*s;
    lambda = lambda/(prod(lambda)^(1/3));
    
    normK = 2;
    cut = [size(obj.subR,1), size(obj.subR,1)+size(obj.gradu,1)];
    if isempty(obj.var)
        obj.var.x = zeros([size(K, 2) 1]);
        obj.var.y = zeros([size(K, 1) 1]);
    else
        if any(size(obj.var.x) - [size(K, 2), 1])
            obj.var.x = zeros([size(K, 2), 1]);
        end
        if any(size(obj.var.y) - [size(K, 1), 1])
            obj.var.y = zeros([size(K, 1), 1]);
        end
    end
    if isempty(obj.prevvar)
        obj.prevvar = obj.var;
    end
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-8,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = lambda(3)*obj.solverOpts.sinoW(:); iW = lambda(2)*obj.solverOpts.imW(:);
    
    k=0;inner_loop = 100;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        
        % Primal iteration: just a threshold
        obj.var.x = max(0, obj.var.x-s*prevKStary);
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut(1))+obj.subR*xbar;
        y2 = reshape(obj.var.y(cut(1)+1:cut(2))+obj.gradu*xbar,[],2);
        y3 = reshape(obj.var.y(cut(2)+1:end)+D(xbar),[],2);
        
        y1 = (y1-t*obj.f)/(1+t/lambda(1));
        y2 = normProj(y2,iW);
        y3 = normProj(y3,sW);
        obj.var.y = [y1(:);y2(:);y3(:)];
        
        KStary = Kconj*obj.var.y(:);
        
       
        % Adaptive step size:
        d = (2/gamma)*s*t*(obj.var.x-obj.prevvar.x)'*(KStary-prevKStary)/...
            (t*norm(obj.var.x-obj.prevvar.x)^2 + s*norm(obj.var.y-obj.prevvar.y)^2);
        if d > 1
            s = s*delta/d;
            t = t*delta/d;
        end


        k = k+1;
        prevKStary = KStary;
    end

        % Primal-Dual Gap calculation:
        p = [sum((lambda(1)/2)*(obj.subR*obj.var.x-obj.f).^2) + sum(iW.*norms(reshape(obj.gradu*obj.var.x,[],2),2,2))+ sum(sW.*norms(reshape(D(obj.var.x),[],2),2,2)),0];
        d = [sum(y1.*((1/(2*lambda(1)))*y1 + obj.f)) , max(0,-min(KStary))];
        pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
%             ip_gap = sum(obj.var.x.*KStary)-sum(obj.var.y.*(K*obj.var.x));
%             disp(['ip_gap: ', num2str(abs(ip_gap))]);
%             fprintf('%f %f %f %f\n',sum((obj.subR*obj.var.x(:)-obj.f(:)).^2)/2,lambda^2*sum(norms(reshape(obj.gradu*obj.var.x(:),[],2),2,2)),(1/2)*sum(y1(:).^2) + sum(y1(:).*obj.f(:)),-min(KStary))
%             fprintf('%f %f %f %f\n', p(1), d(1), p(1)+d(1), pd_gap(1));
        end
        pd_gap = max(abs(pd_gap));

        pd_gap = [...
            (lambda(1)/2)*sum((obj.subR*obj.var.x-obj.f).^2)+sum(y1.^2./(2*lambda(1))+y1.*obj.f)-y1'*(obj.subR*obj.var.x),...
            sum(iW.*norms(reshape(obj.gradu*obj.var.x,size(y2)),2,2))-y2(:)'*(obj.gradu*obj.var.x),...
            sum(sW.*norms(reshape(D(obj.var.x),size(y3)),2,2))-y3(:)'*D(obj.var.x),...
            (KStary'*obj.var.x-y1'*(obj.subR*obj.var.x)-y2(:)'*(obj.gradu*obj.var.x)-y3(:)'*D(obj.var.x))...
            ];
        disp(log10(abs(pd_gap)))
        pd_gap = sum(abs(pd_gap(1:end-1)))/p(1);
            
        % Sensitivity calculation:
        normX = norm(obj.var.x,inf);normY = norm(obj.var.y,inf);
        if normX > eps && normY > eps
            sensitivity = norm(obj.var.x - obj.prevvar.x,inf)/normX ...
                + norm(obj.var.y - obj.prevvar.y,inf)/normY;
        else
            sensitivity = Inf;
        end
    
    end
    
    if sensitivity <= tol(1)
        disp(['Change between iterations dropped below sensitivity barrier after ', num2str(k), ' iterations'])
    end
    if pd_gap <= tol(2)
        disp(['Primal dual gap converged after ', num2str(k), ' iterations'])
    else
        disp(['Final primal-dual gap: ', num2str(pd_gap)])
    end
    

end

function Symmetric_PDHG(obj, lambda, quiet)
%% TV + DTV reconstruction
% % Minimise \frac{\lambda_1}{2}\|Ru-[Mv,(1-M)f]\|^2 + \lambda_2\|Du\|_{2,1} 
%           + \frac{\lambda_3}{2}\|Mv-f\|^2 + \lambda_4\|Jv\|_{2,1}
% % We split this into:
% % $F(x) = \frac{\lambda_1}{2}\|x(1)-(1-M)f\|^2 + \lambda_2\|x(2)\|_{2,1}+ \lambda_4\|x(3)\|_{2,1}$
% % $G(u,v) = \chi_{u\geq 0} + \frac{\lambda_3}{2}\|Mv-f\|^2$
% % $F^*(y) = \frac{1}{2\lambda_1}\|y(1)\|^2 + <(1-M)y(1),f> + \chi_{|y(i)|_2\leq\lambda_i}$
% % $G^*(z) = \chi_{z(1)\leq 0} + \frac{1}{2\lambda_3}\|Mz(2)\|^2 + <Mz(2),f> + \chi_{(1-M)z(2)=0}$
% % $prox_{tF^*}(Y) = [(Y(1))/(1+t/\lambda_1), Y(i)/max(1,|Y(i)|_2/\lambda_i)]$
% % $prox_{sG}(U,V) = [max(0,U),M(V+sf)/(1+s\lambda_3)]$
% L/2|Rx-(1-M)f|^2 + 1/(2L)|y|^2 +<(1-M)y,f> = <Rx,y>
% y/L + (1-M)f = Rx, y = L(Rx-f)

    im_sz = size(obj.fullR,2);
    cut = [size(obj.fullR,1), size(obj.fullR,1)+size(obj.gradu,1)];
    M = true(obj.sino_sz); M(:,obj.angle_mask) = false;
    scale = [1/3,1];%10*[1/3,1e3];
    K = fcthdlop([sum(size(obj.fullR)),1],[size(obj.fullR,1)+size(obj.gradu,1)+size(obj.diffop,1),1],...
        @(x)[obj.fullR*x(1:im_sz)-M(:).*x(1+im_sz:end);obj.gradu*(x(1:im_sz)*scale(1));obj.diffop*(x(1+im_sz:end)*scale(2))], ...
        @(y)[obj.fullR'*y(1:cut(1))+(obj.gradu'*y(1+cut(1):cut(2)))*scale(1);(obj.diffop'*y(cut(2)+1:end))*scale(2)-M(:).*y(1:cut(1))]);
    Kconj = K';

    normK = 1;
    not_angles = setdiff(1:obj.sino_sz(1),obj.angle_mask);
    if isempty(obj.var)
        obj.var.x = zeros([size(K, 2) 1]);
        obj.var.y = zeros([size(K, 1) 1]);
    else
        if any(size(obj.var.x) - [size(K, 2) 1])
            obj.var.x = zeros([size(K, 2) 1]);
        end
        if any(size(obj.var.y) - [size(K, 1), 1])
            obj.var.y = zeros([size(K, 1) 1]);
        end
    end
    if isempty(obj.prevvar)
        obj.prevvar = obj.var;
    end
%     y = K*obj.var.x;
%     y1 = y(1:cut(1)); y1(~M) = (y1(~M)-obj.f)/lambda(1);
%     y2 = reshape(y(cut(1)+1:cut(2)),[],2); y2 = (lambda(2)/scale(1))*y2./repmat(max(eps,norms(y2,2,2)),1,2);
%     y3 = reshape(y(cut(2)+1:end),[],2); y3 = (lambda(4)/scale(2))*y3./repmat(max(eps,norms(y3,2,2)),1,2);
%     obj.var.y = [y1(:);y2(:);y3(:)];obj.prevvar.y = obj.var.y;
    
    
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-6,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = lambda(1)*obj.solverOpts.sinoW(:);
    
    k=0;inner_loop = 100;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        % Primal iteration: just a threshold
        obj.var.x = obj.var.x-s*prevKStary;
        v = reshape(obj.var.x(im_sz+1:end),obj.sino_sz);
        if lambda(3) == inf
            v(~M) = obj.f;
        else
            v(~M) = (v(~M)+(lambda(3)*s)*obj.f)/(1+lambda(3)*s);
        end
        obj.var.x = [max(0, obj.var.x(1:im_sz));v(:)];
        xbar1 = t*(2*obj.var.x(1:im_sz) - obj.prevvar.x(1:im_sz));
        xbar2 = reshape(t*(2*obj.var.x(im_sz+1:end) - obj.prevvar.x(im_sz+1:end)),obj.sino_sz);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut(1))+obj.fullR*xbar1-M(:).*xbar2(:);
        y2 = reshape(obj.var.y(cut(1)+1:cut(2))+obj.gradu*(xbar1*scale(1)),[],2);
        y3 = reshape(obj.var.y(cut(2)+1:end)+obj.diffop*(xbar2(:)*scale(2)),[],2);
        
        y1(~M) = y1(~M)-t*obj.f;
        y1 = y1./(1+t./sW);
        y2 = normProj(y2,lambda(2)/scale(1));
        y3 = normProj(y3,lambda(4)/scale(2));
        obj.var.y = [y1(:);y2(:);y3(:)];
        
        KStary = Kconj*obj.var.y(:);
        
        % Adaptive step size:
        d = (2/gamma)*s*t*(obj.var.x-obj.prevvar.x)'*(KStary-prevKStary)/...
            (t*norm(obj.var.x-obj.prevvar.x)^2 + s*norm(obj.var.y-obj.prevvar.y)^2);
        if d > 1
            s = s*delta/d;
            t = t*delta/d;
        end


        k = k+1;
        prevKStary = KStary;
    end
        
        % Primal-Dual Gap calculation:
        V = v; V(~M) = obj.f;
        p = [(1/2)*sum(sW.*(obj.fullR*obj.var.x(1:im_sz)-V(:)).^2)+lambda(2)*sum(norms(reshape(obj.gradu*obj.var.x(1:im_sz),[],2),2,2))+lambda(4)*sum(norms(reshape(obj.diffop*v(:),[],2),2,2)),0];
        tmp = reshape(-KStary(1+im_sz:end),size(v));
        d = [sum(y1.*y1./(2*sW)), max(0,-min(KStary(1:im_sz)))+max(max(abs(tmp(not_angles,:))))];
        if lambda(3) == inf
            p = p +[0,max(abs(reshape(v(obj.angle_mask,:),[],1)-obj.f))];
            d = d +[sum(reshape(tmp(obj.angle_mask,:),[],1).*obj.f),0];
        else
            p = p +[(lambda(3)/2)*sum((reshape(v(:,obj.angle_mask),[],1)-obj.f).^2),0];
            d = d +[sum(reshape(tmp(:,obj.angle_mask),[],1).*(reshape(tmp(:,obj.angle_mask),[],1)/(2*lambda(3))+obj.f)),0];
        end
        
        pd_gap = [...
            (1/2)*sum(sW.*(obj.fullR*obj.var.x(1:im_sz)-V(:)).^2)+sum(y1.^2./(2*sW)+y1.*V(:))-y1'*(obj.fullR*obj.var.x(1:im_sz)),...
            lambda(2)*sum(norms(reshape(obj.gradu*obj.var.x(1:im_sz),size(y2)),2,2))-y2(:)'*(obj.gradu*obj.var.x(1:im_sz))*scale(1),...
            (lambda(3)/2)*sum((v(~M)-obj.f).^2)+sum(tmp(~M).*(tmp(~M)/(2*lambda(3))+obj.f))-tmp(:)'*v(:),...
            lambda(4)*sum(norms(reshape(obj.diffop*v(:),size(y3)),2,2))-y3(:)'*(obj.diffop*v(:))*scale(2),...
            ];
        disp(log10(abs(pd_gap)))
        pd_gap = sum(abs(pd_gap(1:end)))/p(1);
        
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(pd_gap), '  Infeasibility: ', num2str(p(2) + d(2))])
        end
        pd_gap = max(pd_gap,p(2)+d(2));

        % Sensitivity calculation:
        normX = norm(obj.var.x,inf);normY = norm(obj.var.y,inf);
        if normX > eps && normY > eps
            sensitivity = norm(obj.var.x - obj.prevvar.x,inf)/normX ...
                + norm(obj.var.y - obj.prevvar.y,inf)/normY;
        else
            sensitivity = Inf;
        end
    end
    
    if sensitivity <= tol(1)
        disp(['Change between iterations dropped below sensitivity barrier after ', num2str(k), ' iterations'])
    end
    if pd_gap <= tol(2)
        disp(['Primal dual gap converged after ', num2str(k), ' iterations'])
    else
        disp(['Final primal-dual gap: ', num2str(pd_gap)])
    end
    

end


function y = normProj(x,r)
    y = zeros(size(x));
    aux = sqrt(x(:,1).^2+x(:,2).^2); ind = aux>r;
    if numel(r) == 1
        aux(ind) = r./aux(ind);
    else
        aux(ind) = r(ind)./aux(ind);
    end
    aux(~ind) = 1;
    y(:,1) = x(:,1).*aux;y(:,2) = x(:,2).*aux;

end