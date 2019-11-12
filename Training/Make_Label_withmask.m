
load('Temp1.mat');
load('Temp2.mat');
load('Temp3.mat');
load('Temp4.mat');
load('Temp5.mat');

savepath='train.h5';
savepath2='train2.h5';
savepath3='train3.h5';

size_input = 40;
size_label = 40;
stride = 20;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
data2 = zeros(size_input, size_input, 4, 1);
label2 = zeros(size_label, size_label, 1, 1);
data3 = zeros(size_input, size_input, 1, 1);
label3 = zeros(size_label, size_label, 1, 1);

mask_nums=442;

Mask=zeros(size_input, size_input, mask_nums);
count = 0;

for nums=1: mask_nums   
    mask_name=['mask/mask_', num2str(nums), '.tif'];
    Mask(:, :, nums)=imread(mask_name);
end

Mask=double(Mask)./255;

[w, h]=size(Temp1);

for x = 1 : stride : w-size_input+1
    for y = 1 :stride : h-size_input+1
        
        count=count+1;
        subim_label =  Temp1( x : x+size_label-1, y : y+size_label-1);
        subim_input = Temp1(x : x+size_input-1, y : y+size_input-1) .* Mask(:, :, mod(count, mask_nums)+1);
        
        T2_Linear = Temporal_Linear_Fit(subim_input, Temp2(x : x+size_input-1, y : y+size_input-1), Mask(:, :, mod(count, mask_nums)+1));
        T3_Linear = Temporal_Linear_Fit(subim_input, Temp3(x : x+size_input-1, y : y+size_input-1), Mask(:, :, mod(count, mask_nums)+1));
        T4_Linear = Temporal_Linear_Fit(subim_input, Temp4(x : x+size_input-1, y : y+size_input-1), Mask(:, :, mod(count, mask_nums)+1));
        T5_Linear = Temporal_Linear_Fit(subim_input, Temp5(x : x+size_input-1, y : y+size_input-1), Mask(:, :, mod(count, mask_nums)+1));
        
        subim_input2(:, :, 1, :) = T2_Linear;
        subim_input2(:, :, 2, :) = T3_Linear;
        subim_input2(:, :, 3, :) = T4_Linear;
        subim_input2(:, :, 4, :) = T5_Linear;
        
        subim_label2 =  Temp1( x : x+size_label-1, y : y+size_label-1);
        
        subim_input3=1-Mask(:, :, mod(count, mask_nums)+1);
        subim_label3=1-Mask(:, :, mod(count, mask_nums)+1);
        
        data(:, :, :, count) = subim_input;
        label(:, :, :, count) = subim_label;
        data2(:, :, :, count) = subim_input2;
        label2(:, :, :, count) = subim_label2;
        data3(:, :, :, count) = subim_input3;
        label3(:, :, :, count) = subim_label3;       
        
    end
end

disp(count);
disp('OK 0 ');
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);
data2 = data2(:, :, :, order);
label2 = label2(:, :, :, order);
data3 = data3(:, :, :, order);
label3 = label3(:, :, :, order);
disp('OK 1 ');

%% writing to HDF5
chunksz =128;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    
    batchdata2 = data2(:,:,:,last_read+1:last_read+chunksz);
    batchlabs2 = label2(:,:,:,last_read+1:last_read+chunksz);
    
    batchdata3 = data3(:,:,:,last_read+1:last_read+chunksz);
    batchlabs3 = label3(:,:,:,last_read+1:last_read+chunksz);    
    
    startloc   = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    startloc2  = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    startloc3  = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    
    curr_dat_sz   = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    curr_dat_sz2  = store2hdf5_2(savepath2, batchdata2, batchlabs2, ~created_flag, startloc2, chunksz);
    curr_dat_sz3  = store2hdf5_3(savepath3, batchdata3, batchlabs3, ~created_flag, startloc3, chunksz);
    
    created_flag = true;
    totalct = curr_dat_sz(end);
    
    disp(batchno*128);
end

disp('OK 2 ');
h5disp(savepath);
h5disp(savepath2);
h5disp(savepath3);