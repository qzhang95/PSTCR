function [Temp_2_linear]= Temporal_Linear_Fit(Cloud_1, Temp_2, Mask)

clean_num=find(Mask==1);
x=zeros(size(clean_num));
y=zeros(size(clean_num));
[w, h]=size(Cloud_1);

%search non-cloud data
i=0;
for j=1: w
    for k=1: h
        
        if(Mask(j, k)~=0)
            i=i+1;
            y(i)=Cloud_1(j, k);
            x(i)=Temp_2(j, k);
        end
        
    end
end

% linear fit
par=polyfit(x, y, 1);
Temp_2_linear=par(1).*Temp_2+par(2);

