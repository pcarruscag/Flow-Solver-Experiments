fid = fopen("msh3Dderiv.dat","r");

Nstream = fread(fid,1,'int32');
Npitch  = fread(fid,1,'int32');
Nspan   = fread(fid,1,'int32');
Nvars   = fread(fid,1,'int32');

data = fread(fid,Nstream*Npitch*Nspan*Nvars,'single');
fclose(fid);
data = reshape(data,numel(data)/Nvars,Nvars);

layer = 5;
var = 26;

offset = Nstream*Npitch;
layerData = data((1:offset)+(layer-1)*offset,var);

layerData = reshape(layerData,Nstream,Npitch)';

contourf(layerData)
colorbar
