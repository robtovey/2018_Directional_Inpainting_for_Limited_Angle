classdef solver_GD < solver
%   Solver to compute 
%   $$ \min_{u\geq0} \frac12\|SRu-f|^2 + \alpha\|D\nabla Ru\|_{2,1} + \beta\|\nabla u|_{2,1}$$

    methods(Access=public)
        function obj = solver_GD(varargin)
            obj@solver(varargin{:});
        end
        function solve(obj,alpha,beta, v,D,uk,T,quiet)
            if nargin < 8
                quiet = false;
            end
            lambda = [1,alpha,beta];
%             lambda = lambda/(prod(lambda)^(1/3));
%             log10(lambda)

            if isnumeric(D)
                GD_PDHG(obj, lambda, v(:), D(:), uk(:), T, quiet);
            elseif numel(D) == 2
                GD_PDHG_b(obj, lambda, v(:), D, uk(:), T, quiet);
            else
                lambda = lambda/(prod(lambda)^(1/3));
                GD_PDHG_c(obj, lambda, v(:), uk(:), T, quiet);
            end
        end
    end
    
end


function GD_PDHG_c(obj, lambda,v,uk,T, quiet)
%% TV + DTV reconstruction
% % Minimise \frac{\lambda_1}{2}\|Ru-v\|^2 + \lambda_2T|u-uk|_1 + \lambda_3\|Du\|_{2,1} 
% % We split this into:
% % $F(x) = \frac{\lambda_1}{2}\|x(1)-v\|^2 + \lambda_3\|x(2)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0} + \lambda_2T|u-uk|_1$
% % $F^*(y) = \frac{1}{2\lambda_1}\|y(1)\|^2 +<y(1),v>+ \chi_{|y(2)|_2\leq\lambda_3}$
% % $G^*(z) = \infty*(z>\lambda_2T) -\lambda_2Tvu_k*(z<-\lambda_2T) +vu_k*(|z|<\lambda_2T)$
% % $prox_{tF^*}(Y) = [(Y(1)-t*v)/(1+t/\lambda_1), Y(2)/max(1,|Y(2)|_2/\lambda_3)]$
% % $prox_{sG}(U) = max(0,(U-uk-s\lambda_2Tsign(U-uk))*(|U-uk|>s\lambda_2T) + uk)$

    cut = size(obj.fullR,1);
    K = [obj.fullR;obj.gradu];
    Kconj = K';

    normK = sqrt(2);
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
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-6,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = obj.solverOpts.sinoW(:).^2;iW = obj.solverOpts.imW(:);
    T = T*iW(:);
    
%     fig = figure;
    k=0;inner_loop = 1000;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        % Primal iteration: just a threshold
        obj.var.x = obj.var.x-s*prevKStary-uk;
        obj.var.x = (obj.var.x-(lambda(2)*s*T).*sign(obj.var.x)).*(abs(obj.var.x)>lambda(2)*s*T) + uk;
        obj.var.x = max(0, obj.var.x);
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut)+obj.fullR*xbar;
        y2 = reshape(obj.var.y(cut+1:end)+obj.gradu*xbar,[],2);
        
        y1 = (y1-t*v)./(1+t./(lambda(1)*sW));
        y2 = normProj(y2,lambda(3));
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
%         fig;
%         subplot(2,2,1);imagesc(reshape(obj.var.x,obj.recon_sz));
%         title(['Iteration ' num2str(k)]);
%         subplot(2,2,2);imagesc(reshape(obj.fullR*obj.var.x,obj.sino_sz)');title('Ru');
%         subplot(2,2,3);imagesc(reshape(v,obj.sino_sz)');title('v');colorbar;
%         subplot(2,2,4);imagesc(reshape(abs(v-obj.fullR*obj.var.x),obj.sino_sz)');title('|v-Ru|');
%         drawnow
        
        % Primal-Dual Gap calculation:
        tmp = -KStary./(lambda(2)*T);
        p = [(lambda(1)/2)*sum(sW.*(obj.fullR*obj.var.x-v).^2)+lambda(2)*sqrt(sum(T.*(obj.var.x-uk).^2))+lambda(3)*sum(norms(reshape(obj.gradu*obj.var.x,[],2),2,2)),0];
        d = [sum(y1.^2./(2*lambda(1)*sW)+y1.*v) +sum(-(lambda(2)*T).^2.*tmp.*uk.*(tmp<-1) + lambda(2)*T.*tmp.*uk.*(abs(tmp)<=1)), max(0,max(tmp)-1)];
        pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
        
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
%             ip_gap = sum(obj.var.x.*KStary)-sum(obj.var.y.*(K*obj.var.x));
%             disp(['ip_gap: ', num2str(abs(ip_gap)/(abs(p(1)) + abs(d(1))+10*sqrt(eps)))]);
%             fprintf('%f %f %f %f\n',sum((obj.subR*obj.var.x(:)-obj.f(:)).^2)/2,lambda^2*sum(norms(reshape(obj.gradu*obj.var.x(:),[],2),2,2)),(1/2)*sum(y1(:).^2) + sum(y1(:).*obj.f(:)),-min(KStary))
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
    
%     close(fig);pause(0.1);

end


function GD_PDHG_b(obj, lambda,v,D,uk,T, quiet)
%% TV + DTV reconstruction
% % Minimise \frac{\lambda_1}{2}\|Ru-v\|^2 + \lambda_2(\|a+Bu\|_{2,1}+T|u-uk|_{2,1}^2) + \lambda_3\|Du\|_{2,1} 
% % We split this into:
% % $F(x) = \frac{\lambda_1}{2}\|x(1)-v\|^2 +\lambda_2\|a+x(2)\|_{2,1}+ \lambda_3\|x(3)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0} + \lambda_2T|u-uk|^2$
% % $F^*(y) = \frac{1}{2\lambda_1}\|y(1)\|^2 +<y(1),v>+ \chi_{|y(2)|_2\leq\lambda_2}-<a,y(2)> +\chi_{|y(3)|_2\leq\lambda_3}$
% % $G^*(z) = <uk,z> + (|z|_{2,1}^2-|z|_{2,2}^2)/(2*\lambda_2*T)
% % $prox_{tF^*}(Y) = [(Y(1)-t*v)/(1+t/\lambda_1), (Y(2)+a/t)/max(1,|Y(2)+a/t|_2/\lambda_2), Y(3)/max(1,|Y(3)|_2/\lambda_3)]$
% % $prox_{sG}(U) = max(0,(U+2\lambda_2sTuk)/(1+2\lambda_2sT))$
    T = T*lambda(2);
    a = reshape(D{1},[],2); B = D{2}*obj.fullR;
    tmp = normest(B)/10;
    lambda(2) = lambda(2)*tmp; a = a/tmp;
    B = fcthdlop([size(B,2),1],[size(B,1),1],@(x)D{2}*(obj.fullR*(x/tmp)),@(x)(obj.fullR'*(D{2}'*x))/tmp);
    cut = [size(obj.fullR,1),size(obj.fullR,1)+size(B,1)];
    K = [obj.fullR;B;obj.gradu];
    Kconj = K';
    
    normK = 1;%sqrt(2+tmp^2); %sqrt(2+normest(D{2})^2);
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
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-6,1e-5]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = obj.solverOpts.sinoW(:).^2;iW = obj.solverOpts.imW(:);
    a = bsxfun(@times,a,iW(:));T = T*iW(:);
    
%     y1 = lambda(1)*sW.*(obj.fullR*obj.var.x-v);
%     y2 = a + reshape(B*obj.var.x,[],2); r = max(eps,norms(y2,2,2)); y2 = y2./[r,r];
%     y3 = reshape(obj.gradu*obj.var.x,[],2); r = max(eps,norms(y3,2,2)); y3 = y3./[r,r];
%     obj.var.y = [y1(:);y2(:);y3(:)];
    
    prevKStary = Kconj*obj.var.y(:);
    
%     fig = figure;
    k=0;inner_loop = 100;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        % Primal iteration: just a threshold
        obj.var.x = obj.var.x-s*prevKStary;
        obj.var.x = max(0, (obj.var.x+(2*s*T).*uk)./(1+2*s*T));
%         obj.var.x = (obj.var.x+(2*s*T).*uk)./(1+2*s*T);
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut(1))+obj.fullR*xbar;
        y2 = reshape(obj.var.y(cut(1)+1:cut(2))+B*xbar,[],2);
        y3 = reshape(obj.var.y(cut(2)+1:end)+obj.gradu*xbar,[],2);
        
        y1 = (y1-t*v)./(1+t./(lambda(1)*sW));
        y2 = normProj(y2+a*t,lambda(2));
        y3 = normProj(y3,lambda(3)*iW(:));
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
        p = [(lambda(1)/2)*sum(sW.*(obj.fullR*obj.var.x-v).^2)+lambda(2)*sum(norms(a+reshape(B*obj.var.x,size(y2)),2,2))+sum(T.*(obj.var.x-uk).^2)+lambda(3)*sum(norms(reshape(obj.gradu*obj.var.x,size(y3)),2,2)),0];
        d = [sum(y1.^2./(2*lambda(1)*sW)+y1.*v)-sum(sum(a.*y2)),0];
        if max(T(:)) > 0
        tmp = max(0,uk-KStary./(2*T));
        d(1) = d(1) + sum(-KStary.*tmp-T.*(tmp-uk).^2);
        end
        pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
        
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
%             ip_gap = sum(obj.var.x.*KStary)-sum(obj.var.y.*(K*obj.var.x));
%             disp(['ip_gap: ', num2str(abs(ip_gap)/(abs(p(1)) + abs(d(1))+10*sqrt(eps)))]);
%             fprintf('%f %f %f %f\n',sum((obj.subR*obj.var.x(:)-obj.f(:)).^2)/2,lambda^2*sum(norms(reshape(obj.gradu*obj.var.x(:),[],2),2,2)),(1/2)*sum(y1(:).^2) + sum(y1(:).*obj.f(:)),-min(KStary))
        end
%         fprintf('%f %f %f %f\n', p(1), d(1), p(1)+d(1), pd_gap(1));
%     y1 = lambda(1)*sW.*(obj.fullR*obj.var.x-v);
%     y2 = a + reshape(B*obj.var.x,[],2); r = max(eps,norms(y2,2,2)); y2 = lambda(2)*iW(:).*y2./[r,r];
%     y3 = reshape(obj.gradu*obj.var.x,[],2); r = max(eps,norms(y3,2,2)); y3 = lambda(3)*iW(:).*y3./[r,r];
    tmp = -KStary;%2*T*(uk-obj.var.x);
    tmp2 = max(0,uk+tmp./(2*T));
        pd_gap = [...
            (lambda(1)/2)*sum(sW.*(obj.fullR*obj.var.x-v).^2)+sum(y1.^2./(2*lambda(1)*sW)+y1.*v)-y1'*(obj.fullR*obj.var.x),...
            lambda(2)*sum(norms(a+reshape(B*obj.var.x,size(y2)),2,2))-sum(sum(a.*y2))-y2(:)'*(B*obj.var.x),...
            sum(T.*(obj.var.x-uk).^2)+sum(tmp.*tmp2-T.*(tmp2-uk).^2)-tmp(:)'*obj.var.x,...
            lambda(3)*sum(norms(reshape(obj.gradu*obj.var.x,size(y3)),2,2))-y3(:)'*(obj.gradu*obj.var.x),...
            (KStary'*obj.var.x-y1'*(obj.fullR*obj.var.x)-y2(:)'*(B*obj.var.x)-y3(:)'*(obj.gradu*obj.var.x))...
            ];
        disp(log10(abs(pd_gap)))
        pd_gap = sum(abs(pd_gap(1:4)))/p(1);
            
%         pd_gap = max(abs(pd_gap));

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
    
%     close(fig);pause(0.1);

end

function GD_PDHG(obj, lambda,v,D,uk,T, quiet)
%% TV + DTV reconstruction
% % Minimise \frac{\lambda_1}{2}\|Ru-v\|^2 + \lambda_2(<D,u>+T|u-uk|^2) + \lambda_3\|Du\|_{2,1} 
% % We split this into:
% % $F(x) = \frac{\lambda_1}{2}\|x(1)-v\|^2 + \lambda_3\|x(2)\|_{2,1}$
% % $G(u) = \chi_{u\geq 0} + \lambda_2(<D,u>+T|u-uk|^2)$
% % $F^*(y) = \frac{1}{2\lambda_1}\|y(1)\|^2 +<y(1),v>+ \chi_{|y(2)|_2\leq\lambda_3}$
% % $G^*(z) = <x,z-\lambda_2D> - \lambda_2T|x-uk|^2 @ x = max(0, z/(2\lambda_2 T)-D/2T+uk)$
% % $prox_{tF^*}(Y) = [(Y(1)-t*v)/(1+t/\lambda_1), Y(2)/max(1,|Y(2)|_2/\lambda_3)]$
% % $prox_{sG}(U) = max(0,(U+2\lambda_2sTuk - \lambda_2sD)/(1+2\lambda_2sT))$

    T=lambda(2)*T; D=lambda(2)*D;

    cut = size(obj.fullR,1);
    K = [obj.fullR;obj.gradu];
    Kconj = K';

    normK = 1;
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
    prevKStary = Kconj*obj.var.y(:);
    
    [s,t] = obj.getStepSizes(normK); tol = [1e-6,1e-4]; sensitivity=1;pd_gap=1;
    gamma = obj.solverOpts.gamma;delta = obj.solverOpts.delta;
    sW = obj.solverOpts.sinoW(:).^2;iW = obj.solverOpts.imW(:);
    
%     fig = figure;
    k=0;inner_loop = 100;
    while k<10 ||((k <= obj.solverOpts.maxiter-inner_loop) &&...
                    (sensitivity > tol(1)) && ...
                    (pd_gap > tol(2)))
    for i=1:inner_loop
        obj.prevvar = obj.var;
        % Primal iteration: just a threshold
        obj.var.x = obj.var.x-s*prevKStary;
        obj.var.x = max(0, (obj.var.x+(2*s*T).*uk - s*D)./(1+2*s*T));
        xbar = t*(2*obj.var.x - obj.prevvar.x);
        
        % Dual iteration: translation and rescaling
        y1 = obj.var.y(1:cut)+obj.fullR*xbar;
        y2 = reshape(obj.var.y(cut+1:end)+obj.gradu*xbar,[],2);
        
        y1 = (y1-t*v)./(1+t./(lambda(1)*sW));
        y2 = normProj(y2,lambda(3)*iW);
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
%         fig;
%         subplot(2,2,1);imagesc(reshape(obj.var.x,obj.recon_sz));
%         title(['Iteration ' num2str(k)]);
%         subplot(2,2,2);imagesc(reshape(obj.fullR*obj.var.x,obj.sino_sz)');title('Ru');
%         subplot(2,2,3);imagesc(reshape(v,obj.sino_sz)');title('v');colorbar;
%         subplot(2,2,4);imagesc(reshape(abs(v-obj.fullR*obj.var.x),obj.sino_sz)');title('|v-Ru|');
%         drawnow
        
        % Primal-Dual Gap calculation:
        tmp = max(0,-KStary./(2*T) - D./(2*T)+uk);
        p = [(lambda(1)/2)*sum(sW.*(obj.fullR*obj.var.x-v).^2)+sum(D.*obj.var.x+T.*(obj.var.x-uk).^2)+lambda(3)*sum(iW.*norms(reshape(obj.gradu*obj.var.x,[],2),2,2)),0];
        d = [sum(y1.^2./(2*lambda(1)*sW)+y1.*v) - sum((KStary+D).*tmp + T.*(tmp-uk).^2), max(0,-min(KStary))];
%         pd_gap = max(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1))+10*sqrt(eps)), p(2) + d(2));
        
        if ~quiet
            disp(['Objective: ', num2str(p(1)), ' Gap: ', num2str(abs(p(1)+d(1))/(abs(p(1)) + abs(d(1)))), '  Infeasibility: ', num2str(p(2) + d(2))])
%             ip_gap = sum(obj.var.x.*KStary)-sum(obj.var.y.*(K*obj.var.x));
%             disp(['ip_gap: ', num2str(abs(ip_gap)/(abs(p(1)) + abs(d(1))+10*sqrt(eps)))]);
%             fprintf('%f %f %f %f\n',sum((obj.subR*obj.var.x(:)-obj.f(:)).^2)/2,lambda^2*sum(norms(reshape(obj.gradu*obj.var.x(:),[],2),2,2)),(1/2)*sum(y1(:).^2) + sum(y1(:).*obj.f(:)),-min(KStary))
%             fprintf('%f %f %f %f\n', p(1), d(1), p(1)+d(1), pd_gap(1));
        end
    tmp = -KStary;
    tmp2 = max(0,uk+tmp./(2*T)-D./(2*T));
        pd_gap = [...
            (lambda(1)/2)*sum(sW.*(obj.fullR*obj.var.x-v).^2)+sum(y1.^2./(2*lambda(1)*sW)+y1.*v)-y1'*(obj.fullR*obj.var.x),...
            sum(T.*(obj.var.x-uk).^2+D.*obj.var.x)+sum(tmp2.*(tmp-D)-T.*(tmp2-uk).^2)-tmp(:)'*obj.var.x,...
            lambda(3)*sum(iW.*norms(reshape(obj.gradu*obj.var.x,size(y2)),2,2))-y2(:)'*(obj.gradu*obj.var.x),...
            (KStary'*obj.var.x-y1'*(obj.fullR*obj.var.x)-y2(:)'*(obj.gradu*obj.var.x))...
            ];
        disp(log10(abs(pd_gap)))
        pd_gap = sum(abs(pd_gap(1:3)))/p(1);
        

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
    
%     close(fig);pause(0.1);

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