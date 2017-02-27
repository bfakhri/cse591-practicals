% License: MIT
% (c) 2017 Ragav Venkatesan
%
% Code to convert a (downlaoded from online) dataset to proper format
% usable by yann to setup the dataset in its own internal format.
% As seen here in this folder. (These are processed mats.
%
% Download test_32x32.mat, train_32x32.mat and extra_32x32.mat from 
% http://ufldl.stanford.edu/housenumbers/ website, format 2.
% Then run this code. 
%
% batch_sizing and number of batches are parameters that can be 
% set in the second stage of code.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% Percent of Data I want to store, as a decimal
tol_batches = 2

load('test_32x32.mat');
x_test = shiftdim(X,3);
els = floor(size(x_test, 1)*(tol_batches/10));
x_test = x_test(1:els,:);
x_test = double(x_test);
y_test = y(1:els, :);
clear X;
clear y;

load('train_32x32.mat');
x_train = shiftdim(X,3);
els = floor(size(x_train, 1)*(tol_batches/10));
x_train = x_train(1:els,:);
x_train = double(x_train);
y_train = y(1:els,:);
clear X;
clear y;

load('extra_32x32.mat');
x_valid = shiftdim(X,3);
els = floor(size(x_valid, 1)*(tol_batches/10));
x_valid = x_valid(1:els,:);
x_valid = double(x_valid);
y_valid = y(1:els,:);
clear X;
clear y;

x = [x_train; x_test; x_valid];
y = [y_train; y_test; y_valid];

clearvars -except x y tol_batches
% save ('data.mat' , '-v7.3');
mkdir('train');
mkdir('test');
mkdir('valid');

%% 
% Going to throw away 420 samples.
throw_away = 420; 
batch_size = 500;

data = x (1:length(x) - throw_away,:);
labels = y (1:length(y) - throw_away) - 1; % because labels go from 1-10

total_batches = length(labels) / batch_size;
test_size = floor(total_batches / 3);
remain = total_batches - test_size; 

train_size = floor(2* remain / 3);
remain = remain - train_size;
valid_size = remain; 

clear x
clear y;
%% 
x = data(  1:train_size * batch_size ,:);
y = labels(1:train_size * batch_size);
dump( 'train',tol_batches, batch_size, x, y );

x = data(  train_size * batch_size + 1 : train_size * batch_size + test_size * batch_size ,:);
y = labels(train_size * batch_size + 1 : train_size * batch_size + test_size * batch_size);
dump( 'test',tol_batches, batch_size, x, y );

x = data(  (train_size + test_size) * batch_size + 1 : end ,:);
y = labels((train_size + test_size) * batch_size + 1 : end);
dump( 'valid',tol_batches, batch_size, x, y );