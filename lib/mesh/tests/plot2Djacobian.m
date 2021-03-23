fid = fopen("msh2Dderiv.dat")

Nstream = fread(fid,1,'int32');
Npitch  = fread(fid,1,'int32');
Nlayers = fread(fid,1,'int32');
Nvars   = 2*Nstream;

data = fread(fid,Nstream*Npitch*Nlayers*Nvars,'single');
fclose(fid);
data = reshape(data,numel(data)/Nlayers,Nlayers);

for i=1:Nvars
  for j=1:Nlayers
    subplot(1,Nlayers,j)
    J = data((1:Npitch*Nstream)+(i-1)*Npitch*Nstream,j);
    J = reshape(J,Nstream,Npitch)';
    contourf(J)
  end
  pause(0.1)
end