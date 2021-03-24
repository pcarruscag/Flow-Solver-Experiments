task = "blade_to_blade";
variable = "velocity";
jLE = 28;
jTE = 84;
Nspan = 53;
Npitch = 61;
offset = 4;
color = ['r','g','b'];

load("mesh.txt")
x = mesh(:,1)*0.001;
y = mesh(:,2)*0.001;
z = mesh(:,3)*0.001;
clear mesh
Nmeri = numel(x)/(Nspan*Npitch);

results = load("flow.txt");
v = sqrt(results(:,1).^2+results(:,2).^2+results(:,3).^2);
p = results(:,4);
k = results(:,5);
w = results(:,6);
clear results

x = reshape(x,Nmeri,Npitch,Nspan);
y = reshape(y,Nmeri,Npitch,Nspan);
z = reshape(z,Nmeri,Npitch,Nspan);
v = reshape(v,Nmeri,Npitch,Nspan);
p = reshape(p,Nmeri,Npitch,Nspan);
k = reshape(k,Nmeri,Npitch,Nspan);
w = reshape(w,Nmeri,Npitch,Nspan);

i=0;
for section = [1+offset round((Nspan-1)/2+1) Nspan-offset]
  i=i+1;
  
  teta = squeeze(atan2(y(:,:,section),x(:,:,section)));
  ri = squeeze(sqrt(x(:,:,section).^2+y(:,:,section).^2));
  zi = squeeze(z(:,:,section));
  
  stream = zeros(Nmeri,Npitch);
  for j=2:Nmeri
    stream(j,:) = stream(j-1,:)+...
                  sqrt((ri(j,:)-ri(j-1,:)).^2+(zi(j,:)-zi(j-1,:)).^2);
  end
  %stream = stream./repmat(stream(end,:),Nmeri,1);
  
  if strcmp(task,"blade_to_blade")
    if strcmp(variable,"velocity")
      data = squeeze(v(:,:,section));
    elseif strcmp(variable,"pressure")
      data = squeeze(p(:,:,section));
    elseif strcmp(variable,"k")
      data = squeeze(k(:,:,section));
    elseif strcmp(variable,"omega")
      data = squeeze(w(:,:,section));
    end
    subplot(1,3,i),contourf(teta.*ri,stream,data,32)
    axis equal
    colorbar()
  elseif strcmp(task,"blade_loading")
    plot(stream(jLE:jTE,  1   ),squeeze(p(jLE:jTE,  1   ,section)),color(i),...
         stream(jLE:jTE,Npitch),squeeze(p(jLE:jTE,Npitch,section)),color(i))
    hold on
  end
end
hold off

if strcmp(task,"inlet_to_outlet")
  if strcmp(variable,"velocity")
    data = v;
  elseif strcmp(variable,"pressure")
    data = p;
  elseif strcmp(variable,"k")
    data = k;
  elseif strcmp(variable,"omega")
    data = w;
  end
  for i=1:Nmeri
    contourf(squeeze(y(i,:,:)),squeeze(x(i,:,:)),squeeze(data(i,:,:)));
    colorbar();
    pause(0.25)
  end
end
