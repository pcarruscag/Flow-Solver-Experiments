load("msh2D.txt");

nLayers = 3;

[M,N] = size(msh2D);

M /= 2*nLayers;

clf
for k=1:nLayers
  subplot(1,nLayers,k)
  hold on;
  for i=1:M
    plot(msh2D((k-1)*2*M+i,:),msh2D((k-0.5)*2*M+i,:))
  end

  for i=1:N
    plot(msh2D((1:M)+(k-1)*2*M,i),msh2D((1:M)+(k-0.5)*2*M,i))
  end
  axis("equal")
  hold off;
end
