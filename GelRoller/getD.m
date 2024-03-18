clear;clc;

theta = 61.0876;

Rl = 35.5;

Theta = zeros(640, 1);
for i=1:640
    Theta(i) = (i - 321) * theta / 640;
end

H = zeros(640, 1);
for i=1:640
    H(i) = Rl * (1 - cosd(Theta(i)));
end

depth = zeros(480, 640);
for i=1:480
    depth(i, :) = H';
end

rect = [92, 108, 448, 373]; % crop region: [xmin ymin width height]
depth_crop = imcrop(depth, rect);
save('D.mat', "depth_crop");