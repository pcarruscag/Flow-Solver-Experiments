msh3D = load("msh3D2.txt");

Nblades = 9;
Nspan = 20;
Npitch = 30;
Nstream = numel(msh3D)/3/Nspan/Npitch;
offset = Nstream*Npitch;

k = 20;

xCoords = msh3D((1:offset)+(k-1)*offset,1);
yCoords = msh3D((1:offset)+(k-1)*offset,2);
zCoords = msh3D((1:offset)+(k-1)*offset,3);

xCoords = reshape(xCoords,Nstream,Npitch)';
yCoords = reshape(yCoords,Nstream,Npitch)';
zCoords = reshape(zCoords,Nstream,Npitch)';

clf
mesh(zCoords,yCoords,xCoords);
hold on

teta = 2*pi/Nblades;
R = [cos(teta) -sin(teta) 0;
     sin(teta)  cos(teta) 0;
        0          0      1];
coords2 = msh3D((1:offset)+(k-1)*offset,:)*R;

xCoords = reshape(coords2(:,1),Nstream,Npitch)';
yCoords = reshape(coords2(:,2),Nstream,Npitch)';
zCoords = reshape(coords2(:,3),Nstream,Npitch)';

mesh(zCoords,yCoords,xCoords);
axis("equal")
hold off