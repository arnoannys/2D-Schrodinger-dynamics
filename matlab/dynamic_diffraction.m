%% ARNO ANNYS

%% diffraction pattern from slit experiment %%
% single slit
% double slit
% dirichlet B.C.
clear
clc
%% Specifying parameters%%
N = 40 ;                          
NN = N^2 ;
nx= N;                           %Number of steps in space(x)
ny=N;                            %Number of steps in space(y)       
nt=160;                          %Number of time steps 
dt= 0.001;                       %Width of each time step
dx=2/(nx-1);                     %Width of space step(x)
dy=2/(ny-1);                     %Width of space step(y)
h = dx ;
x=0:dx:2;                        %Range of x(0,2) and specifying the grid points
y=0:dy:2;                        %Range of y(0,2) and specifying the grid points
u=zeros(nx^2,ny^2);              %Preallocating 
psin = zeros(NN,1) ;             %Preallocating 
L = 1000 ; 
i_imag = sqrt(-1) ; 
%% initial wave function %%
x0 = 1;  
y0 = 0.5;                        % Location of the center of the wavepacket
velocity = 40;                   % Average velocity of the packet
k0 =   velocity;                 % Average wavenumber
sigmax0 = L/4000; 
sigmay0 = L/4000 ;               % Standard deviation of the wavefunction
Norm_psi =  1/(sqrt(sigmax0*sqrt(pi)));                   % Normalization
[xx, yy] = meshgrid(x, y);
u = Norm_psi*exp(i_imag*k0*xx).* ...
    exp(-((xx'-x0).^2/(2*sigmax0^2)+(yy'-y0).^2/(2*sigmay0^2)));    %gaussian pulse
psi = reshape(u,[NN,1]) ;        %creating vector for computation

%% high potential barrier (avoid tunneling) with slits; enable 1 %%
%%single slit %
% ggrid = zeros(N) ;
% [xgrid,ygrid] = meshgrid(x , y) ;
% ggrid(:,35) = ones(N,1) ;
% ggrid(17:23,35) = 0 ;             
% mesh(xgrid , ygrid , ggrid) ;
% title('visual of the setup')
% height = 100000;
% M = reshape(ggrid, [NN,1]) ;

% double slit %
ggrid = zeros(N) ;
[xgrid,ygrid] = meshgrid(x , y) ;
ggrid(:,30) = ones(N,1) ;
ggrid(16:19,30) = 0 ;                 
ggrid(21:24,30) = 0 ;                  
mesh(xgrid , ygrid , ggrid) ;
title('visual of the setup')
height = 100000;
M = reshape(ggrid, [NN,1]) ;

pot = height*spdiags( M , [0] , NN,NN) ;

%% initializing hamiltonian %%
coeff = -1/(2*h^2);              % Co�ffici�nt for Hamiltoniaan
ham = coeff*spdiags( [ ones(NN,1) ones(NN,1)  -4*ones(NN,1)  ones(NN,1) ones(NN,1)], [-N -1 0 1 N], NN , NN) + pot;

%% initialiing Q (sparse) %%
Q = 1/2*(spdiags([ones(NN,1)], [0], NN , NN) + i_imag*dt/2*ham) ; 

%% Calculating for each time step %%
for it=0:nt
    psin=psi; 
    figure(2)
    P = psi.*conj(psi) ;                       %Probability density
    PP = reshape(P,[N,N]);                     %Probability density vector 
    control = sum(P) 
    %this value will be displayed in the command  window, normalization
    % requires this value to stay constant in time. This can be used as to check stability
    % the algorithm.
                                               
    F=surf(x,y,PP,'EdgeColor','none');       %plotting 
    shading interp
    axis([0 2 0 2 0 5])
    title({['2-D pulse '];['time (\itt) = ',num2str(it*dt)]})
    xlabel('(x) \rightarrow')
    ylabel('{\leftarrow}  (y)')
    zlabel('(prob denst) \rightarrow')
    %view(2) 
    drawnow; 
    refreshdata(F)
    chi = Q\psi ;
    psin =  chi - psi ; 
    psi = psin ;                              %calculate new value and replace
    
    % plotting diffraction pattern (at one peak moment)
    if it == 0.091/dt
    figure(3)
    pattern=surf(x,y,PP,'EdgeColor','none');       %plotting 
    shading interp
    axis ([1.999 2 0 2 0 5]) 
   
    title({['diffraction pattern'];['time (\itt) = ',num2str(it*dt)]})
    zlabel('(prob denst) \rightarrow')
    view(90,0)
    end

end