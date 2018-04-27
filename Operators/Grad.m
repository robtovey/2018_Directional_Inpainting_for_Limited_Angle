classdef Grad < fcthdlop
    properties(GetAccess = protected, SetAccess = protected)
        sz        % size of input
        ndim      % dimension of output
        normVals  % values of 1-norm, 2-norm and inf-norm
    end
    methods(Access = public)
        function obj = Grad(sz,normalise)
            if nargin == 1
                normalise = true;
            end
            ndim = numel(sz);
            if sz(end) == 1
                ndim = ndim - 1;
            end
            
            if normalise
                scale = 1/sqrt(4*ndim);
                normVals = [2*ndim,1/scale,2]*scale;
                switch ndim
                    case 1
                        fwrd = @(z)fw1d(reshape(z,sz)*scale);
                        bwrd = @(z)bw1d(reshape(z,[sz,ndim]))*scale;
                    case 2
                        fwrd = @(z)fw2d(reshape(z,sz)*scale);
                        bwrd = @(z)bw2d(reshape(z,[sz,ndim]))*scale;
                    case 3
                        fwrd = @(z)fw3d(reshape(z,sz)*scale);
                        bwrd = @(z)bw3d(reshape(z,[sz,ndim]))*scale;
                    case 4
                        fwrd = @(z)fw4d(reshape(z,sz)*scale);
                        bwrd = @(z)bw4d(reshape(z,[sz,ndim]))*scale;
                    otherwise
                        error(['ndim = ' num2str(ndim) 'is not currently supported'])
                end
            else
                normVals = [2*ndim,1,2];
                switch ndim
                    case 1
                        fwrd = @(z)fw1d(reshape(z,sz));
                        bwrd = @(z)bw1d(reshape(z,[sz,ndim]));
                    case 2
                        fwrd = @(z)fw2d(reshape(z,sz));
                        bwrd = @(z)bw2d(reshape(z,[sz,ndim]));
                    case 3
                        fwrd = @(z)fw3d(reshape(z,sz));
                        bwrd = @(z)bw3d(reshape(z,[sz,ndim]));
                    case 4
                        fwrd = @(z)fw4d(reshape(z,sz));
                        bwrd = @(z)bw4d(reshape(z,[sz,ndim]));
                    otherwise
                        error(['ndim = ' num2str(ndim) ' is not currently supported'])
                end
            end
            obj@fcthdlop(sz, [sz ndim], fwrd, bwrd);
            
            obj.sz = sz; obj.ndim = ndim; obj.normVals = normVals;
            
        end
        
        function n = norm(obj,p)
            if nargin == 1
                p = 2;
            end
            
            switch p
                case 1
                    n = obj.normVals(1);
                case 2
                    n = obj.normVals(2);
                case inf
                    n = obj.normVals(3);
                otherwise
                    error(['p-norm of grad is not known for p = ' num2str(p)]);
            end
        end
    end
end

%% The gradient operators
% The forwards operator is the gradient and the backwards operator
% -divergence. Note that the slight assymetry between dimensions is caused
% as we would like grad = [dx, dy, dz,...] whereas matlab's indices are
% ordered [y,x,z,...].

function grad = fw1d(u)
% Forward differences with Neumann boundary
grad = zeros([size(u) 1]);
grad(1:(end - 1)) = u(2:end) - u(1:(end - 1));
end
function u = bw1d(grad)
% Backward differences with Dirichlet boundary
dim = size(grad);
u = zeros(dim(1),1);
u(1) = u(1) - grad(1);
u(2:(end - 1)) = u(2:(end - 1)) + grad(1:(end - 2))...
    - grad(2:(end - 1));
u(end) = u(end) + grad(end - 1);
end

function grad = fw2d(u)
% Forward differences with Neumann boundary
grad = zeros([size(u) 2]);
grad(:, 1:(end - 1), 1) = u(:, 2:end) - u(:, 1:(end - 1));
grad(1:(end - 1), :, 2) = u(2:end, :) - u(1:(end - 1), :);
end
function u = bw2d(grad)
% Backward differences with Dirichlet boundary
dim = size(grad);
u = zeros(dim(1:2));
u(:, 1) = -grad(:, 1, 1);
u(:, 2:(end - 1)) = grad(:, 1:(end - 2), 1) - grad(:, 2:(end - 1), 1);
u(:, end) = grad(:, end - 1, 1);

u(1, :) = u(1, :) - grad(1, :, 2);
u(2:(end - 1), :) = u(2:(end - 1), :) + grad(1:(end - 2), :, 2)...
    - grad(2:(end - 1), :, 2);
u(end, :) = u(end, :) + grad(end - 1, :, 2);
end

function grad = fw3d(u)
% Forward differences with Neumann boundary
grad = zeros([size(u) 3]);
grad(:, 1:(end - 1), :, 1) = u(:, 2:end, :) - u(:, 1:(end - 1), :);
grad(1:(end - 1), :, :, 2) = u(2:end, :, :) - u(1:(end - 1), :, :);
grad(:, :, 1:(end - 1), 3) = u(:, :, 2:end) - u(:, :, 1:(end - 1));
end
function u = bw3d(grad)
% Backward differences with Dirichlet boundary
dim = size(grad);
u = zeros(dim(1:3));
u(:, 1, :) = -grad(:, 1, :, 1);
u(:, 2:(end - 1), :) = grad(:, 1:(end - 2), :, 1) - grad(:, 2:(end - 1),:, 1);
u(:, end, :) = grad(:, end - 1, :, 1);

u(1, :, :) = u(1, :, :) - grad(1, :, :, 2);
u(2:(end - 1), :, :) = u(2:(end - 1), :, :) + grad(1:(end - 2), :, :, 2)...
    - grad(2:(end - 1), :, :, 2);
u(end, :, :) = u(end, :, :) + grad(end - 1, :, :, 2);

u(:, :, 1) = u(:, :, 1) - grad(:, :, 1, 3);
u(:, :, 2:(end - 1)) = u(:, :, 2:(end - 1)) + grad(:, :, 1:(end - 2), 3)...
    - grad(:, :, 2:(end - 1), 3);
u(:, :, end) = u(:, :, end) + grad(:, :, end - 1, 3);

end

function grad = fw4d(u)
% Forward differences with Neumann boundary
grad = zeros([size(u) 4]);
grad(:, 1:(end - 1), :, :, 1) = u(:, 2:end, :, :) - u(:, 1:(end - 1), :, :);
grad(1:(end - 1), :, :, :, 2) = u(2:end, :, :, :) - u(1:(end - 1), :, :, :);
grad(:, :, 1:(end - 1), :, 3) = u(:, :, 2:end, :) - u(:, :, 1:(end - 1), :);
grad(:, :, :, 1:(end - 1), 4) = u(:, :, :, 2:end) - u(:, :, :, 1:(end - 1));
end
function u = bw4d(grad)
% Backward differences with Dirichlet boundary
dim = size(grad);
u = zeros(dim(1:4));
u(:, 1, :, :) = -grad(:, 1, :, :, 1);
u(:, 2:(end - 1), :, :) = grad(:, 1:(end - 2), :, :, 1) - grad(:, 2:(end - 1), :, :, 1);
u(:, end, :, :) = grad(:, end - 1, :, :, 1);

u(1, :, :, :) = u(1, :, :, :) - grad(1, :, :, :, 2);
u(2:(end - 1), :, :, :) = u(2:(end - 1), :, :, :) + grad(1:(end - 2), :, :, :, 2)...
    - grad(2:(end - 1), :, :, :, 2);
u(end, :, :, :) = u(end, :, :, :) + grad(end - 1, :, :, :, 2);

u(:, :, 1, :) = u(:, :, 1, :) - grad(:, :, 1, :, 3);
u(:, :, 2:(end - 1), :) = u(:, :, 2:(end - 1), :) + grad(:, :, 1:(end - 2), :, 3)...
    - grad(:, :, 2:(end - 1), :, 3);
u(:, :, end, :) = u(:, :, end, :) + grad(:, :, end - 1, :, 3);

u(:, :, :, 1) = u(:, :, :, 1) - grad(:, :, :, 1, 4);
u(:, :, :, 2:(end - 1)) = u(:, :, :, 2:(end - 1)) + grad(:, :, :, 1:(end - 2), 4)...
    - grad(:, :, :, 2:(end - 1), 4);
u(:, :, :, end) = u(:, :, :, end) + grad(:, :, :, end - 1, 4);

end