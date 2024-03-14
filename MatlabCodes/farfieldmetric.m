%% setup scaling parameters
clear
addpath('src')
addpath('results')

N = 80;
h =  1/(N-1);
%% setup the model and the domain

% background wavespeed
c = ones(N,N);
 
% size of the model in interior domain
nxi  = size(c,2);
nyi  = size(c,1);
ni   = nxi*nyi;

xi = h*(0:nxi-1) - 0.5;
yi = h*(0:nyi-1) - 0.5;

[Xi,Yi] = meshgrid(xi,yi);

% size of the simulation domain
npml = 20;
nx = nxi + 2*npml;
ny = nyi + 2*npml;

x  = [xi(1)+(-npml:-1)*h xi xi(end)+(1:npml)*h];
y  = [yi(1)+(-npml:-1)*h yi yi(end)+(1:npml)*h];

[X,Y] = meshgrid(x,y);

% order of accuracy
order = 8;

% intensity of the pml absorbtion
sigmaMax = 80;


%%
etas = h5read('shepplogan.hdf5', '/eta');
eta_predicted = h5read('shepplogan.hdf5', '/eta_predicted');


size(etas)
size(eta_predicted) 
%%
m = 4;
n = 20;
Ntheta = N;
% the number of angles of the sources and detectors (they are set to be the same).
dtheta = 2*pi/(Ntheta);

theta = linspace(0, 2*pi-dtheta, Ntheta);
d = [cos(theta).' sin(theta).'];

theta_r = linspace(0, 2*pi-dtheta, Ntheta);
r = [cos(theta_r).' sin(theta_r).'];

points_query = 0.5*r;
project_mat = zeros(Ntheta, nx, ny);

for ii = 1:nx
    for jj = 1:ny
        mat_dummy = zeros(nx,ny);
        mat_dummy(ii,jj) = 1;
        project_mat(:,ii,jj) = interp2(x,y,...
                                   reshape(mat_dummy, nx, ny),...
                                   points_query(:,1),...
                                   points_query(:,2));
    end
end

% properly reshaping and making it sparse
project_mat = sparse(reshape(project_mat, Ntheta, nx*ny));
%%
scatter_real_freq_5 = zeros(Ntheta^2, n, m);
scatter_imag_freq_5 = zeros(Ntheta^2, n, m);
scatter_real_freq_10 = zeros(Ntheta^2, n, m);
scatter_imag_freq_10 = zeros(Ntheta^2, n, m);
scatter_real_freq_20 = zeros(Ntheta^2, n, m);
scatter_imag_freq_20 = zeros(Ntheta^2, n, m);

%%

omega5 = 2*2.5*pi;
omega10 = 2*5*pi;
omega20 = 2*10*pi;

U_in5 =  exp(1i*omega5*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in10 =  exp(1i*omega10*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));
U_in20 =  exp(1i*omega20*(X(:)*d(:,1).'+ Y(:)*d(:,2).'));

%%
for j = 1:m
for i = 1:n
    i+20*(j-1)
    eta = eta_predicted(1,:,:,i,j);
    meta = 1 + eta;  
    eta_ext = ExtendModel(eta,nxi,nyi,npml);
    mext = ExtendModel(meta,nxi,nyi,npml);
   
    H5 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega5,'compact_explicit');
    H10 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega10,'compact_explicit');
    H20 = HelmholtzMatrix(mext,nx,ny,npml,h,...
        sigmaMax,order,omega20,'compact_explicit');
    
    % building the right hand sides
    S5 = bsxfun(@times, -omega5^2*eta_ext, U_in5);
    S10 = bsxfun(@times, -omega10^2*eta_ext, U_in10);
    S20 = bsxfun(@times, -omega20^2*eta_ext, U_in20);
    
    % solving the equation 
    U5 = H5\S5;
    U10 = H10\S10;
    U20 = H20\S20;

    % this is our "real data"
    scatter5 = project_mat*U5;
    scatter10 = project_mat*U10;
    scatter20 = project_mat*U20;
    
%     % adding noise
%     scatter = scatter.*(1 + randn(nxi,nyi));
    scatter_real_freq_5(:,i, j) = real(reshape(scatter5, Ntheta^2, 1));
    scatter_imag_freq_5(:,i, j) = imag(reshape(scatter5, Ntheta^2, 1));
    scatter_real_freq_10(:,i, j) = real(reshape(scatter10, Ntheta^2, 1));
    scatter_imag_freq_10(:,i, j) = imag(reshape(scatter10, Ntheta^2, 1));
    scatter_real_freq_20(:,i, j) = real(reshape(scatter20, Ntheta^2, 1));
    scatter_imag_freq_20(:,i, j) = imag(reshape(scatter20, Ntheta^2, 1));
end
end
%%
exact_real_freq_5 = zeros(Ntheta^2, m);
exact_imag_freq_5 = zeros(Ntheta^2, m);
exact_real_freq_10 = zeros(Ntheta^2, m);
exact_imag_freq_10 = zeros(Ntheta^2, m);
exact_real_freq_20 = zeros(Ntheta^2, m);
exact_imag_freq_20 = zeros(Ntheta^2, m);

%%
for j = 1:m
j
eta = etas(1,:,:,j);
meta = 1 + eta;  
eta_ext = ExtendModel(eta,nxi,nyi,npml);
mext = ExtendModel(meta,nxi,nyi,npml);

H5 = HelmholtzMatrix(mext,nx,ny,npml,h,...
    sigmaMax,order,omega5,'compact_explicit');
H10 = HelmholtzMatrix(mext,nx,ny,npml,h,...
    sigmaMax,order,omega10,'compact_explicit');
H20 = HelmholtzMatrix(mext,nx,ny,npml,h,...
    sigmaMax,order,omega20,'compact_explicit');

% building the right hand sides
S5 = bsxfun(@times, -omega5^2*eta_ext, U_in5);
S10 = bsxfun(@times, -omega10^2*eta_ext, U_in10);
S20 = bsxfun(@times, -omega20^2*eta_ext, U_in20);

% solving the equation 
U5 = H5\S5;
U10 = H10\S10;
U20 = H20\S20;

% this is our "real data"
scatter5 = project_mat*U5;
scatter10 = project_mat*U10;
scatter20 = project_mat*U20;

% adding noise
exact_real_freq_5(:,j) = real(reshape(scatter5, Ntheta^2, 1));
exact_imag_freq_5(:,j) = imag(reshape(scatter5, Ntheta^2, 1));
exact_real_freq_10(:,j) = real(reshape(scatter10, Ntheta^2, 1));
exact_imag_freq_10(:,j) = imag(reshape(scatter10, Ntheta^2, 1));
exact_real_freq_20(:,j) = real(reshape(scatter20, Ntheta^2, 1));
exact_imag_freq_20(:,j) = imag(reshape(scatter20, Ntheta^2, 1));
end
%%
error_real_freq_5  = zeros(n,m);
error_imag_freq_5  = zeros(n,m);
error_real_freq_10 = zeros(n,m);
error_imag_freq_10 = zeros(n,m);
error_real_freq_20 = zeros(n,m);
error_imag_freq_20 = zeros(n,m);

for j = 1:m
for i = 1:n
    i+20*(j-1)
    error_real_freq_5(i,j)  = norm(scatter_real_freq_5(:,i,j)-exact_real_freq_5(:,j))/norm(exact_real_freq_5(:,j));
    error_imag_freq_5(i,j)  = norm(scatter_imag_freq_5(:,i,j)-exact_imag_freq_5(:,j))/norm(exact_imag_freq_5(:,j));
    error_real_freq_10(i,j) = norm(scatter_real_freq_10(:,i,j)-exact_real_freq_10(:,j))/norm(exact_real_freq_10(:,j));
    error_imag_freq_10(i,j) = norm(scatter_imag_freq_10(:,i,j)-exact_imag_freq_10(:,j))/norm(exact_imag_freq_10(:,j));
    error_real_freq_20(i,j) = norm(scatter_real_freq_20(:,i,j)-exact_real_freq_20(:,j))/norm(exact_real_freq_20(:,j));
    error_imag_freq_20(i,j) = norm(scatter_imag_freq_20(:,i,j)-exact_imag_freq_20(:,j))/norm(exact_imag_freq_20(:,j));

end
end
%%
mean(mean(error_real_freq_5))
mean(mean(error_imag_freq_5))
mean(mean(error_real_freq_10))
mean(mean(error_imag_freq_10))
mean(mean(error_real_freq_20))
mean(mean(error_imag_freq_20))


