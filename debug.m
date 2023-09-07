close all
clear
clc

% data = load("Contributions_40_0_0.txt");
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens);
% figure(1)
% pcshow(pc), xlabel('X')

% data = load("Contributions_50_0_0.txt");
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens);
% figure(2)
% pcshow(pc), xlabel('X')

data = load("mat_20_0_0.mat");
data = data.data;
X = data(:,1);
Y = data(:,2);
Z = data(:,3);
Intens = data(:,4);
pc = pointCloud([X Y Z], Intensity=Intens);
figure(3)
pcshow(pc), xlabel('X')
max(X), max(Y)
% 
% data = load("mat_50_0_0.mat");
% data = data.data;
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens);
% figure(4)
% pcshow(pc), xlabel('X')
% max(X), max(Y)

% data = load("img_20_0_0.mat");
% data = data.image;
% figure(3)
% imshow(data)
% 
% data = load("img_40_0_0.mat");
% data = data.image;
% figure(4)
% imshow(data)

% data = load("mat_1.mat");
% data = data.data;
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens, Color="g");
% figure(1)
% pcshow(pc), xlabel('X'), hold on
% 
% data = load("mat_2.mat");
% data = data.data;
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens, Color="blue");
% pcshow(pc), hold on
% 
% data = load("mat_3.mat");
% data = data.data;
% X = data(:,1);
% Y = data(:,2);
% Z = data(:,3);
% Intens = data(:,4);
% pc = pointCloud([X Y Z], Intensity=Intens, Color="r");
% pcshow(pc)