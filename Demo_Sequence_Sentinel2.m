
%% Author Information
%------------------------------
% Qiang Zhang
% LIESMARS, Wuhan University
% whuqzhang@gmail.com
% 2019.07.26
%------------------------------

clc;
clear;

%% Model
def  = 'Def/Model.prototxt';
model= 'Def/Model.caffemodel';

caffe.reset_all();
caffe.set_mode_gpu();

%% Loading Temproal Sequence Data
% Sentinel-2 (B11 Data)
load('Data/Temp_1.mat');
load('Data/Temp_2.mat');
load('Data/Temp_3.mat');
load('Data/Temp_4.mat');
load('Data/Temp_5.mat');

%% Simulated Cloudy With Mask
ori_mask_1=double(imread('sequential_masks/mask_1.tif'))./255;
ori_mask_2=double(imread('sequential_masks/mask_2.tif'))./255;
ori_mask_3=double(imread('sequential_masks/mask_4.tif'))./255;
ori_mask_4=double(imread('sequential_masks/mask_3.tif'))./255;
ori_mask_5=double(imread('sequential_masks/mask_5.tif'))./255;

mask_1=ori_mask_1(:, :, 1);
mask_2=ori_mask_2(:, :, 1);
mask_3=ori_mask_3(:, :, 1);
mask_4=ori_mask_4(:, :, 1);
mask_5=ori_mask_5(:, :, 1);

ori_mask_1=mask_1;
ori_mask_2=mask_2;
ori_mask_3=mask_3;
ori_mask_4=mask_4;
ori_mask_5=mask_5;

Cloud_1=Temp_1.*mask_1;
Original_1=Cloud_1;
Cloud_2=Temp_2.*mask_2;
Original_2=Cloud_2;
Cloud_3=Temp_3.*mask_3;
Original_3=Cloud_3;
Cloud_4=Temp_4.*mask_4;
Original_4=Cloud_4;
Cloud_5=Temp_5.*mask_5;
Original_5=Cloud_5;

[w, h]=size(Cloud_1);

%% Setting Parameters
stride=20;
count=0;
patch_weight=1;
original_stride=stride;

res_1=zeros(w, h); res_2=zeros(w, h); res_3=zeros(w, h); res_4=zeros(w, h); res_5=zeros(w, h);
W=zeros(w, h);
Cloud_iter_1=zeros(w, h); Cloud_iter_2=zeros(w, h); Cloud_iter_3=zeros(w, h); Cloud_iter_4=zeros(w, h); Cloud_iter_5=zeros(w, h);
Cloud_final_1=zeros(w, h); Cloud_final_2=zeros(w, h); Cloud_final_3=zeros(w, h); Cloud_final_4=zeros(w, h); Cloud_final_5=zeros(w, h);
Mask_final_1=zeros(w, h); Mask_final_2=zeros(w, h); Mask_final_3=zeros(w, h); Mask_final_4=zeros(w, h); Mask_final_5=zeros(w, h);

%% Temp_1 Recovering
patch=95;
ratio=0.1;
stride=21;

dtxt = textread(def,'%s','delimiter','\n','whitespace','');
fpo = fopen(def, 'w');
linenum = 1;
for l=1:length(dtxt)
    if(linenum == 5)
        fprintf(fpo, 'input_dim: %d\n', patch);       
    elseif(linenum == 6 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 11 )
        fprintf(fpo, 'input_dim: %d\n', patch);        
    elseif(linenum == 12 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 17 )
        fprintf(fpo, 'input_dim: %d\n', patch);   
    elseif(linenum == 18 )
        fprintf(fpo, 'input_dim: %d\n', patch);      
    else
        fprintf(fpo, dtxt{l, 1});
        fprintf(fpo, '\n');
    end
    linenum = linenum+1;
end
fclose(fpo);

net = caffe.Net(def, model, 'test');
All_Temp=zeros(patch, patch, 4);
All_Mask=zeros(patch, patch, 4);
Temp_patch=zeros(patch, patch, 4);

iter_1=0;
last_rest_radio_1=0;

%% Final Restoration
disp('Reconstructing Temp_1: ');
while (size(find(mask_1(:, :)==0), 1)~=0)
    
    iter_1=iter_1+1;
    %iteration once
    for x = 1: stride : w-patch+1
        for y = 1: stride : h-patch+1
            
            % intact numbers of current patch
            intact_numbers=size(find(mask_1(x:x+patch-1, y:y+patch-1)==1), 1);
            
            % patch needn't fill
            if (intact_numbers==patch*patch || intact_numbers<patch*patch*ratio)
                continue;
                
                % patch need fill
            else
                
                All_Temp(:, :, 1)=Original_2(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 2)=Original_3(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 3)=Original_4(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 4)=Original_5(x:x+patch-1, y:y+patch-1);
                
                All_Mask(:, :, 1)=ori_mask_2(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 2)=ori_mask_3(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 3)=ori_mask_4(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 4)=ori_mask_5(x:x+patch-1, y:y+patch-1, 1);
                
                CC_Temp(1)=CC_Value(Cloud_1(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 1).*mask_1(x:x+patch-1, y:y+patch-1));
                CC_Temp(2)=CC_Value(Cloud_1(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 2).*mask_1(x:x+patch-1, y:y+patch-1));
                CC_Temp(3)=CC_Value(Cloud_1(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 3).*mask_1(x:x+patch-1, y:y+patch-1));
                CC_Temp(4)=CC_Value(Cloud_1(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 4).*mask_1(x:x+patch-1, y:y+patch-1));
                
                [cc_max, pos_max]=sort(CC_Temp);           
                Temp_2_linear = Temporal_Linear_Fit2(Cloud_1(x:x+patch-1, y:y+patch-1), All_Temp(:, :, pos_max(4)), mask_1(x:x+patch-1, y:y+patch-1), All_Mask(:, :, pos_max(4)));
                
                Cloud_patch=Cloud_1(x:x+patch-1, y:y+patch-1);
                mask_patch=1-mask_1(x:x+patch-1, y:y+patch-1);
                Temp_patch(:, :, 1)=Temp_2_linear;
                Temp_patch(:, :, 2)=Temp_2_linear;
                Temp_patch(:, :, 3)=Temp_2_linear;
                Temp_patch(:, :, 4)=Temp_2_linear;
                              
                net_output = net.forward({Cloud_patch, Temp_patch, mask_patch});
                output = net_output{1,1};
                
                Res_patch=output.*(1-mask_1(x:x+patch-1, y:y+patch-1)) + Cloud_patch.*mask_1(x:x+patch-1, y:y+patch-1);
                
                % Update Cloud image and Weight
                patch_weight=exp(1.0/(patch*patch - intact_numbers));
                Cloud_iter_1(x:x+patch-1, y:y+patch-1)=Cloud_iter_1(x:x+patch-1, y:y+patch-1)+Res_patch.*patch_weight;
                W(x:x+patch-1, y:y+patch-1) = W(x:x+patch-1, y:y+patch-1) + patch_weight;                
            end          
        end
    end
    
    % Update final image and mask of current iteration
    for i=1: w
        for j=1: h
            
            if (W(i, j)==0)
                Cloud_final_1(i, j)=Cloud_1(i, j);
                Mask_final_1(i, j)= mask_1(i, j);
            else
                Cloud_final_1(i, j)=Cloud_iter_1(i, j)/W(i, j);
                Mask_final_1(i, j)=1;
            end
            
        end
    end
    
    Cloud_1=Cloud_final_1;
    mask_1= Mask_final_1;
    
    Cloud_iter_1=Cloud_iter_1*0;
    W=W*0;
    Cloud_final_1=Cloud_final_1*0;
    Mask_final_1=Mask_final_1*0;
    
    rest_ratio=100 * size(find(mask_1(:, :)==0), 1) / (w*h);
    if(last_rest_radio_1==rest_ratio)
        stride=stride-1;
    end
    
    if(stride<1)
        stride=1;
    end
       
    last_rest_radio_1=rest_ratio;
    disp(['Iteration: ', num2str(iter_1), '. Rest of missing Regions = ', num2str(floor(rest_ratio)), '%']);
    
end

final_result_1=Cloud_1;
[cc_temp1, ssim_temp1, rmse_temp1] = Evaluation_Index(Temp_1, final_result_1);
disp('Temp_1 Finished!');


%% Temp_2 Recovering
patch=108;
ratio=0.1;
stride=21;

dtxt = textread(def,'%s','delimiter','\n','whitespace','');
fpo = fopen(def, 'w');
linenum = 1;
for l=1:length(dtxt)
    if(linenum == 5)
        fprintf(fpo, 'input_dim: %d\n', patch);       
    elseif(linenum == 6 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 11 )
        fprintf(fpo, 'input_dim: %d\n', patch);        
    elseif(linenum == 12 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 17 )
        fprintf(fpo, 'input_dim: %d\n', patch);   
    elseif(linenum == 18 )
        fprintf(fpo, 'input_dim: %d\n', patch);      
    else
        fprintf(fpo, dtxt{l, 1});
        fprintf(fpo, '\n');
    end
    linenum = linenum+1;
end
fclose(fpo);

net = caffe.Net(def, model, 'test');
All_Temp=zeros(patch, patch, 4);
All_Mask=zeros(patch, patch, 4);
Temp_patch=zeros(patch, patch, 4);

iter_2=0;
last_rest_radio_2=0;

%% Final Restoration
disp('Reconstructing Temp_2: ');
while (size(find(mask_2(:, :)==0), 1)~=0)
    
    iter_2=iter_2+1;
    %iteration once
    for x = 1: stride : w-patch+1
        for y = 1: stride : h-patch+1
            
            % intact numbers of current patch
            intact_numbers=size(find(mask_2(x:x+patch-1, y:y+patch-1)==1), 1);
            
            % patch needn't fill
            if (intact_numbers==patch*patch || intact_numbers<patch*patch*ratio)
                continue;
                
                % patch need fill 
            else               
                All_Temp(:, :, 1)=Original_1(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 2)=Original_3(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 3)=Original_4(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 4)=Original_5(x:x+patch-1, y:y+patch-1);
                
                All_Mask(:, :, 1)=ori_mask_1(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 2)=ori_mask_3(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 3)=ori_mask_4(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 4)=ori_mask_5(x:x+patch-1, y:y+patch-1, 1);                
                
                CC_Temp(1)=size(find(All_Mask(:, :, 1)==1), 1);
                CC_Temp(2)=size(find(All_Mask(:, :, 2)==1), 1);
                CC_Temp(3)=size(find(All_Mask(:, :, 3)==1), 1);
                CC_Temp(4)=size(find(All_Mask(:, :, 4)==1), 1);
                
                [cc_max, pos_max]=sort(CC_Temp);           
                Temp_2_linear = Temporal_Linear_Fit2(Cloud_2(x:x+patch-1, y:y+patch-1), All_Temp(:, :, pos_max(4)), mask_2(x:x+patch-1, y:y+patch-1), All_Mask(:, :, pos_max(4)));
                
                Cloud_patch=Cloud_2(x:x+patch-1, y:y+patch-1);
                mask_patch=1-mask_2(x:x+patch-1, y:y+patch-1);
                Temp_patch(:, :, 1)=Temp_2_linear;
                Temp_patch(:, :, 2)=Temp_2_linear;
                Temp_patch(:, :, 3)=Temp_2_linear;
                Temp_patch(:, :, 4)=Temp_2_linear;
                                
                net_output = net.forward({Cloud_patch, Temp_patch, mask_patch});
                output = net_output{1,1};
                
                Res_patch=output.*(1-mask_2(x:x+patch-1, y:y+patch-1)) + Cloud_patch.*mask_2(x:x+patch-1, y:y+patch-1);
                
                % Update Cloud image and Weight
                patch_weight=exp(1.0/(patch*patch - intact_numbers));
                Cloud_iter_2(x:x+patch-1, y:y+patch-1)=Cloud_iter_2(x:x+patch-1, y:y+patch-1)+Res_patch.*patch_weight;
                W(x:x+patch-1, y:y+patch-1) = W(x:x+patch-1, y:y+patch-1) + patch_weight;              
            end            
        end
    end
    
    % Update final image and mask of current iteration
    for i=1: w
        for j=1: h
            
            if (W(i, j)==0)
                Cloud_final_2(i, j)=Cloud_2(i, j);
                Mask_final_2(i, j)= mask_2(i, j);
            else
                Cloud_final_2(i, j)=Cloud_iter_2(i, j)/W(i, j);
                Mask_final_2(i, j)=1;
            end
            
        end
    end
    
    Cloud_2=Cloud_final_2;
    mask_2=  Mask_final_2;
    
    Cloud_iter_2=Cloud_iter_2*0;
    W=W*0;
    Cloud_final_2=Cloud_final_2*0;
    Mask_final_2=Mask_final_2*0;
    
    rest_ratio=100 * size(find(mask_2(:, :)==0), 1) / (w*h);
    if(last_rest_radio_2==rest_ratio)
        stride=stride-1;
    end
    
    if(stride<1)
        stride=1;
    end
    
    
    last_rest_radio_2=rest_ratio;
    disp(['Iteration: ', num2str(iter_2), '. Rest of missing Regions = ', num2str(floor(rest_ratio)), '%']);
    
end

final_result_2=Cloud_2;

[cc_temp2, ssim_temp2, rmse_temp2] = Evaluation_Index(Temp_2, final_result_2);
disp('Temp_2 Finished!');


%% Temp_3 
patch=108;
ratio=0.1;
stride=21;

dtxt = textread(def,'%s','delimiter','\n','whitespace','');
fpo = fopen(def, 'w');
linenum = 1;
for l=1:length(dtxt)
    if(linenum == 5)
        fprintf(fpo, 'input_dim: %d\n', patch);       
    elseif(linenum == 6 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 11 )
        fprintf(fpo, 'input_dim: %d\n', patch);        
    elseif(linenum == 12 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 17 )
        fprintf(fpo, 'input_dim: %d\n', patch);   
    elseif(linenum == 18 )
        fprintf(fpo, 'input_dim: %d\n', patch);      
    else
        fprintf(fpo, dtxt{l, 1});
        fprintf(fpo, '\n');
    end
    linenum = linenum+1;
end
fclose(fpo);

net = caffe.Net(def, model, 'test');
All_Temp=zeros(patch, patch, 4);
All_Mask=zeros(patch, patch, 4);
Temp_patch=zeros(patch, patch, 4);

iter_3=0;
last_rest_radio_3=0;

%% Final Restoration
disp('Reconstructing Temp_3: ');
while (size(find(mask_3(:, :)==0), 1)~=0)
    
    iter_3=iter_3+1;
    %iteration once
    for x = 1: stride : w-patch+1
        for y = 1: stride : h-patch+1
            
            % intact numbers of current patch
            intact_numbers=size(find(mask_3(x:x+patch-1, y:y+patch-1)==1), 1);
            
            % patch needn't fill 
            if (intact_numbers==patch*patch || intact_numbers<patch*patch*ratio)
                continue;
                
                % patch need fill
            else
                
                All_Temp(:, :, 1)=Original_1(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 2)=Original_2(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 3)=Original_4(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 4)=Original_5(x:x+patch-1, y:y+patch-1);
                
                All_Mask(:, :, 1)=ori_mask_1(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 2)=ori_mask_2(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 3)=ori_mask_4(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 4)=ori_mask_5(x:x+patch-1, y:y+patch-1, 1);
                
                CC_Temp(1)=CC_Value(Cloud_3(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 1).*mask_3(x:x+patch-1, y:y+patch-1));
                CC_Temp(2)=CC_Value(Cloud_3(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 2).*mask_3(x:x+patch-1, y:y+patch-1));
                CC_Temp(3)=CC_Value(Cloud_3(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 3).*mask_3(x:x+patch-1, y:y+patch-1));
                CC_Temp(4)=CC_Value(Cloud_3(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 4).*mask_3(x:x+patch-1, y:y+patch-1));
                
                [cc_max, pos_max]=sort(CC_Temp);           
                Temp_2_linear = Temporal_Linear_Fit2(Cloud_3(x:x+patch-1, y:y+patch-1), All_Temp(:, :, pos_max(4)), mask_3(x:x+patch-1, y:y+patch-1), All_Mask(:, :, pos_max(4)));
                
                Cloud_patch=Cloud_3(x:x+patch-1, y:y+patch-1);
                mask_patch=1-mask_3(x:x+patch-1, y:y+patch-1);
                Temp_patch(:, :, 1)=Temp_2_linear;
                Temp_patch(:, :, 2)=Temp_2_linear;
                Temp_patch(:, :, 3)=Temp_2_linear;
                Temp_patch(:, :, 4)=Temp_2_linear;
                            
                net_output = net.forward({Cloud_patch, Temp_patch, mask_patch});
                output = net_output{1,1};
                
                Res_patch=output.*(1-mask_3(x:x+patch-1, y:y+patch-1)) + Cloud_patch.*mask_3(x:x+patch-1, y:y+patch-1);
                
                % Update Cloud image and Weight
                patch_weight=exp(1.0/(patch*patch - intact_numbers));
                Cloud_iter_3(x:x+patch-1, y:y+patch-1)=Cloud_iter_3(x:x+patch-1, y:y+patch-1)+Res_patch.*patch_weight;
                W(x:x+patch-1, y:y+patch-1) = W(x:x+patch-1, y:y+patch-1) + patch_weight;               
            end           
        end
    end
    
    % Update final image and mask of current iteration
    for i=1: w
        for j=1: h
            
            if (W(i, j)==0)
                Cloud_final_3(i, j)=Cloud_3(i, j);
                Mask_final_3(i, j)= mask_3(i, j);
            else
                Cloud_final_3(i, j)=Cloud_iter_3(i, j)/W(i, j);
                Mask_final_3(i, j)=1;
            end
            
        end
    end
    
    Cloud_3=Cloud_final_3;
    mask_3=  Mask_final_3;
    
    Cloud_iter_3=Cloud_iter_3*0;
    W=W*0;
    Cloud_final_3=Cloud_final_3*0;
    Mask_final_3=Mask_final_3*0;
    
    rest_ratio=100 * size(find(mask_3(:, :)==0), 1) / (w*h);
    if(last_rest_radio_3==rest_ratio)
        stride=stride-1;
    end
    
    if(stride<1)
        stride=1;
    end
    
    
    last_rest_radio_3=rest_ratio;
    disp(['Iteration: ', num2str(iter_3), '. Rest of missing Regions = ', num2str(floor(rest_ratio)), '%']);
    
end

final_result_3=Cloud_3;

[cc_temp3, ssim_temp3, rmse_temp3] = Evaluation_Index(Temp_3, final_result_3);
disp('Temp_3 Finished!');


%% Temp_4 Recovering
patch=110;
ratio=0.1;
stride=21;

dtxt = textread(def,'%s','delimiter','\n','whitespace','');
fpo = fopen(def, 'w');
linenum = 1;
for l=1:length(dtxt)
    if(linenum == 5)
        fprintf(fpo, 'input_dim: %d\n', patch);       
    elseif(linenum == 6 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 11 )
        fprintf(fpo, 'input_dim: %d\n', patch);        
    elseif(linenum == 12 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 17 )
        fprintf(fpo, 'input_dim: %d\n', patch);   
    elseif(linenum == 18 )
        fprintf(fpo, 'input_dim: %d\n', patch);      
    else
        fprintf(fpo, dtxt{l, 1});
        fprintf(fpo, '\n');
    end
    linenum = linenum+1;
end
fclose(fpo);

net = caffe.Net(def, model, 'test');
All_Temp=zeros(patch, patch, 4);
All_Mask=zeros(patch, patch, 4);
Temp_patch=zeros(patch, patch, 4);

iter_4=0;
last_rest_radio_4=0;

%% Final Restoration
disp('Reconstructing Temp_4: ');
while (size(find(mask_4(:, :)==0), 1)~=0)
    
    iter_4=iter_4+1;
    %iteration once
    for x = 1: stride : w-patch+1
        for y = 1: stride : h-patch+1
            
            % intact numbers of current patch
            intact_numbers=size(find(mask_4(x:x+patch-1, y:y+patch-1)==1), 1);
            
            % patch needn't fill
            if (intact_numbers==patch*patch || intact_numbers<patch*patch*ratio)
                continue;
                
                % patch need fill
            else
                
                All_Temp(:, :, 1)=Original_1(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 2)=Original_2(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 3)=Original_3(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 4)=Original_5(x:x+patch-1, y:y+patch-1);
                
                All_Mask(:, :, 1)=ori_mask_1(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 2)=ori_mask_2(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 3)=ori_mask_3(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 4)=ori_mask_5(x:x+patch-1, y:y+patch-1, 1);                
                
                CC_Temp(1)=CC_Value(Cloud_4(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 1).*mask_4(x:x+patch-1, y:y+patch-1));
                CC_Temp(2)=CC_Value(Cloud_4(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 2).*mask_4(x:x+patch-1, y:y+patch-1));
                CC_Temp(3)=CC_Value(Cloud_4(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 3).*mask_4(x:x+patch-1, y:y+patch-1));
                CC_Temp(4)=CC_Value(Cloud_4(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 4).*mask_4(x:x+patch-1, y:y+patch-1));
                
                [cc_max, pos_max]=sort(CC_Temp);           
                Temp_2_linear = Temporal_Linear_Fit2(Cloud_4(x:x+patch-1, y:y+patch-1), All_Temp(:, :, pos_max(4)), mask_4(x:x+patch-1, y:y+patch-1), All_Mask(:, :, pos_max(4)));
                
                Cloud_patch=Cloud_4(x:x+patch-1, y:y+patch-1);
                mask_patch=1-mask_4(x:x+patch-1, y:y+patch-1);
                Temp_patch(:, :, 1)=Temp_2_linear;
                Temp_patch(:, :, 2)=Temp_2_linear;
                Temp_patch(:, :, 3)=Temp_2_linear;
                Temp_patch(:, :, 4)=Temp_2_linear;             
                
                net_output = net.forward({Cloud_patch, Temp_patch, mask_patch});
                output = net_output{1,1};
                
                Res_patch=output.*(1-mask_4(x:x+patch-1, y:y+patch-1)) + Cloud_patch.*mask_4(x:x+patch-1, y:y+patch-1);
                
                % Update Cloud image and Weight
                patch_weight=exp(1.0/(patch*patch - intact_numbers));
                Cloud_iter_4(x:x+patch-1, y:y+patch-1)=Cloud_iter_4(x:x+patch-1, y:y+patch-1)+Res_patch.*patch_weight;
                W(x:x+patch-1, y:y+patch-1) = W(x:x+patch-1, y:y+patch-1) + patch_weight;                
            end           
        end
    end
    
    % Update final image and mask of current iteration
    for i=1: w
        for j=1: h
            
            if (W(i, j)==0)
                Cloud_final_4(i, j)=Cloud_4(i, j);
                Mask_final_4(i, j)= mask_4(i, j);
            else
                Cloud_final_4(i, j)=Cloud_iter_4(i, j)/W(i, j);
                Mask_final_4(i, j)=1;
            end
            
        end
    end
    
    Cloud_4=Cloud_final_4;
    mask_4=  Mask_final_4;
    
    Cloud_iter_4=Cloud_iter_4*0;
    W=W*0;
    Cloud_final_4=Cloud_final_4*0;
    Mask_final_4=Mask_final_4*0;
    
    rest_ratio=100 * size(find(mask_4(:, :)==0), 1) / (w*h);
    if(last_rest_radio_4==rest_ratio)
        stride=stride-1;
    end
    
    if(stride<1)
        stride=1;
    end
      
    last_rest_radio_4=rest_ratio;
    disp(['Iteration: ', num2str(iter_4), '. Rest of missing Regions = ', num2str(floor(rest_ratio)), '%']);
    
end

final_result_4=Cloud_4;
[cc_temp4, ssim_temp4, rmse_temp4] = Evaluation_Index(Temp_4, final_result_4);
disp('Temp_4 Finished!');


%% Temp_5 Recovering
patch=110;
ratio=0.1;
stride=10;

dtxt = textread(def,'%s','delimiter','\n','whitespace','');
fpo = fopen(def, 'w');
linenum = 1;
for l=1:length(dtxt)
    if(linenum == 5)
        fprintf(fpo, 'input_dim: %d\n', patch);       
    elseif(linenum == 6 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 11 )
        fprintf(fpo, 'input_dim: %d\n', patch);        
    elseif(linenum == 12 )
        fprintf(fpo, 'input_dim: %d\n', patch);
    elseif(linenum == 17 )
        fprintf(fpo, 'input_dim: %d\n', patch);   
    elseif(linenum == 18 )
        fprintf(fpo, 'input_dim: %d\n', patch);      
    else
        fprintf(fpo, dtxt{l, 1});
        fprintf(fpo, '\n');
    end
    linenum = linenum+1;
end
fclose(fpo);

net = caffe.Net(def, model, 'test');
All_Temp=zeros(patch, patch, 4);
All_Mask=zeros(patch, patch, 4);
Temp_patch=zeros(patch, patch, 4);

iter_5=0;
last_rest_radio_5=0;

%% Final Restoration
disp('Reconstructing Temp_5: ');
while (size(find(mask_5(:, :)==0), 1)~=0)
    
    iter_5=iter_5+1;
    %iteration once
    for x = 1: stride : w-patch+1
        for y = 1: stride : h-patch+1
            
            % intact numbers of current patch
            intact_numbers=size(find(mask_5(x:x+patch-1, y:y+patch-1)==1), 1);
            
            % patch needn't fill 
            if (intact_numbers==patch*patch || intact_numbers<patch*patch*ratio)
                continue;
                
                % patch need fill
            else
                
                All_Temp(:, :, 1)=Original_1(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 2)=Original_2(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 3)=Original_3(x:x+patch-1, y:y+patch-1);
                All_Temp(:, :, 4)=Original_4(x:x+patch-1, y:y+patch-1);
                
                All_Mask(:, :, 1)=ori_mask_1(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 2)=ori_mask_2(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 3)=ori_mask_3(x:x+patch-1, y:y+patch-1, 1);
				All_Mask(:, :, 4)=ori_mask_4(x:x+patch-1, y:y+patch-1, 1);
                
                CC_Temp(1)=CC_Value(Cloud_5(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 1).*mask_5(x:x+patch-1, y:y+patch-1));
                CC_Temp(2)=CC_Value(Cloud_5(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 2).*mask_5(x:x+patch-1, y:y+patch-1));
                CC_Temp(3)=CC_Value(Cloud_5(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 3).*mask_5(x:x+patch-1, y:y+patch-1));
                CC_Temp(4)=CC_Value(Cloud_5(x:x+patch-1, y:y+patch-1), All_Temp(:, :, 4).*mask_5(x:x+patch-1, y:y+patch-1));
                
                [cc_max, pos_max]=sort(CC_Temp);           
                Temp_2_linear = Temporal_Linear_Fit2(Cloud_5(x:x+patch-1, y:y+patch-1), All_Temp(:, :, pos_max(4)), mask_5(x:x+patch-1, y:y+patch-1), All_Mask(:, :, pos_max(4)));
                
                Cloud_patch=Cloud_5(x:x+patch-1, y:y+patch-1);
                mask_patch=1-mask_5(x:x+patch-1, y:y+patch-1);
                Temp_patch(:, :, 1)=Temp_2_linear;
                Temp_patch(:, :, 2)=Temp_2_linear;
                Temp_patch(:, :, 3)=Temp_2_linear;
                Temp_patch(:, :, 4)=Temp_2_linear;
                                
                net_output = net.forward({Cloud_patch, Temp_patch, mask_patch});
                output = net_output{1,1};
                
                Res_patch=output.*(1-mask_5(x:x+patch-1, y:y+patch-1)) + Cloud_patch.*mask_5(x:x+patch-1, y:y+patch-1);
                
                % Update Cloud image and Weight
                patch_weight=exp(1.0/(patch*patch - intact_numbers));
                Cloud_iter_5(x:x+patch-1, y:y+patch-1)=Cloud_iter_5(x:x+patch-1, y:y+patch-1)+Res_patch.*patch_weight;
                W(x:x+patch-1, y:y+patch-1) = W(x:x+patch-1, y:y+patch-1) + patch_weight;
                
            end           
        end
    end
    
    % Update final image and mask of current iteration
    for i=1: w
        for j=1: h
            
            if (W(i, j)==0)
                Cloud_final_5(i, j)=Cloud_5(i, j);
                Mask_final_5(i, j)= mask_5(i, j);
            else
                Cloud_final_5(i, j)=Cloud_iter_5(i, j)/W(i, j);
                Mask_final_5(i, j)=1;
            end
            
        end
    end
    
    Cloud_5=Cloud_final_5;
    mask_5=  Mask_final_5;
    
    Cloud_iter_5=Cloud_iter_5*0;
    W=W*0;
    Cloud_final_5=Cloud_final_5*0;
    Mask_final_5=Mask_final_5*0;
    
    rest_ratio=100 * size(find(mask_5(:, :)==0), 1) / (w*h);
    if(last_rest_radio_5==rest_ratio)
        stride=stride-1;
    end
    
    if(stride<1)
        stride=1;
    end
    
    
    last_rest_radio_5=rest_ratio;
    disp(['Iteration: ', num2str(iter_5), '. Rest of missing Regions = ', num2str(floor(rest_ratio)), '%']);
    
end

final_result_5=Cloud_5;

[cc_temp5, ssim_temp5, rmse_temp5] = Evaluation_Index(Temp_5, final_result_5);
disp('Temp_5 Finished!');

subplot(3, 5, 1)
imshow(Original_1);
title('Cloudy 1');
subplot(3, 5, 2)
imshow(Original_2);
title('Cloudy 2');
subplot(3, 5, 3)
imshow(Original_3);
title('Cloudy 3');
subplot(3, 5, 4)
imshow(Original_4);
title('Cloudy 4');
subplot(3, 5, 5)
imshow(Original_5);
title('Cloudy 5');

subplot(3, 5, 6)
imshow(final_result_1);
title(['CC: ', num2str(cc_temp1), '  SSIM: ', num2str(ssim_temp1), '  RMSE: ', num2str(rmse_temp1)]);
subplot(3, 5, 7)
imshow(final_result_2);
title(['CC: ', num2str(cc_temp2), '  SSIM: ', num2str(ssim_temp2), '  RMSE: ', num2str(rmse_temp2)]);
subplot(3, 5, 8)
imshow(final_result_3);
title(['CC: ', num2str(cc_temp3), '  SSIM: ', num2str(ssim_temp3), '  RMSE: ', num2str(rmse_temp3)]);
subplot(3, 5, 9)
imshow(final_result_4);
title(['CC: ', num2str(cc_temp4), '  SSIM: ', num2str(ssim_temp4), '  RMSE: ', num2str(rmse_temp4)]);
subplot(3, 5, 10)
imshow(final_result_5);
title(['CC: ', num2str(cc_temp5), '  SSIM: ', num2str(ssim_temp5), '  RMSE: ', num2str(rmse_temp5)]);

subplot(3, 5, 11)
imshow(Temp_1);
title('Original Temp 1');
subplot(3, 5, 12)
imshow(Temp_2);
title('Original Temp 2');
subplot(3, 5, 13)
imshow(Temp_3);
title('Original Temp 3');
subplot(3, 5, 14)
imshow(Temp_4);
title('Original Temp 4');
subplot(3, 5, 15)
imshow(Temp_5);
title('Original Temp 5');

