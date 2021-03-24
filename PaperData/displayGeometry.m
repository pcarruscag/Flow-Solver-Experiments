load("mesh.txt")

x = mesh(:,1);
y = mesh(:,2);
z = mesh(:,3);
clear mesh

Nblade = 9;
jLE = 28;
jTE = 84;
Nspan = 53;
Npitch = 61;
Nmeri = numel(x)/(Nspan*Npitch);

x = reshape(x,Nmeri,Npitch,Nspan);
y = reshape(y,Nmeri,Npitch,Nspan);
z = -reshape(z,Nmeri,Npitch,Nspan);

clf
hold on

function [u,v,w] = myRotate(mat,x,y,z)
  sizes = size(x);
  
  X = [reshape(x,numel(x),1)...
       reshape(y,numel(y),1)...
       reshape(z,numel(z),1)] * mat;
       
  u = reshape(X(:,1),sizes);
  v = reshape(X(:,2),sizes);
  w = reshape(X(:,3),sizes);
endfunction

for iBlade = 1:Nblade

  theta = 2*pi/Nblade * (iBlade-1);
  
  mat = [cos(theta) sin(theta) 0;
        -sin(theta) cos(theta) 0;
         0          0          1];

  x_p1 = squeeze(x(:,:,1));
  y_p1 = squeeze(y(:,:,1));
  z_p1 = squeeze(z(:,:,1));
  
  [x_p1,y_p1,z_p1] = myRotate(mat,x_p1,y_p1,z_p1);
  mesh(x_p1,y_p1,z_p1)

  x_p1 = squeeze(x(jLE:jTE,1,:));
  y_p1 = squeeze(y(jLE:jTE,1,:));
  z_p1 = squeeze(z(jLE:jTE,1,:));
  
  [x_p1,y_p1,z_p1] = myRotate(mat,x_p1,y_p1,z_p1);
  mesh(x_p1,y_p1,z_p1)

  x_p1 = squeeze(x(jLE:jTE,Npitch,:));
  y_p1 = squeeze(y(jLE:jTE,Npitch,:));
  z_p1 = squeeze(z(jLE:jTE,Npitch,:));
  
  [x_p1,y_p1,z_p1] = myRotate(mat,x_p1,y_p1,z_p1);
  mesh(x_p1,y_p1,z_p1)
  
endfor

axis equal
colormap(gray)
caxis([900 1000])

