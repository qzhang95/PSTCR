function cc=CC_Value(A, B)

[n,m]=size(A);

temp1=0.0; temp2=0.0; temp3=0.0;
A_mean=mean2(A);
B_mean=mean2(B);

for i=1:n
    for j=1:m
        
        temp1=double(A(i,j)-A_mean)*double(B(i,j)-B_mean)+temp1; 
        temp2=double(A(i,j)-A_mean)*double(A(i,j)-A_mean)+temp2;
        temp3=double(B(i,j)-B_mean)*double(B(i,j)-B_mean)+temp3;
    end
end

if (temp2*temp3~=0)
    cc=temp1/sqrt(temp2*temp3);
else
    cc=-2;
end

end