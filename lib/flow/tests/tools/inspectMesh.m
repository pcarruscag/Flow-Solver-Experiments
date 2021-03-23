load("mesh.txt")

x = mesh(:,1);
y = mesh(:,2);
z = mesh(:,3);
clear mesh

Nspan = 49;
Npitch = 53;
Nmeri = numel(x)/(Nspan*Npitch);

x = reshape(x,Nmeri,Npitch,Nspan);
y = reshape(y,Nmeri,Npitch,Nspan);
z = reshape(z,Nmeri,Npitch,Nspan);

clf
hold on

mesh(x(:,:,1),y(:,:,1),z(:,:,1))
mesh(x(:,:,Nspan),y(:,:,Nspan),z(:,:,Nspan))

x_p1 = squeeze(x(:,1,:));
y_p1 = squeeze(y(:,1,:));
z_p1 = squeeze(z(:,1,:));

mesh(x_p1,y_p1,z_p1)

x_p1 = squeeze(x(:,Npitch,:));
y_p1 = squeeze(y(:,Npitch,:));
z_p1 = squeeze(z(:,Npitch,:));

mesh(x_p1,y_p1,z_p1)

x_p1 = squeeze(x(1,:,:));
y_p1 = squeeze(y(1,:,:));
z_p1 = squeeze(z(1,:,:));

mesh(x_p1,y_p1,z_p1)

clear x_p1 y_p1 z_p1