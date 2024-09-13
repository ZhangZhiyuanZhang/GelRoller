clear;clc;
height = 480;
width = 640;

theta = 61.0876;

Theta = zeros(height, width);

for c = 1:width
    Theta(:, c) = (c - 321) * theta / 640;
end

Normal = zeros(height, width, 3);
for r = 1:height
    for c = 1:width
        Normal(r, c, 1) = -sind(Theta(r, c));
        Normal(r, c, 2) = 0.0;
        Normal(r, c, 3) = -cosd(Theta(r, c));
    end
end

rect = [92, 108, 448, 373]; % crop region: [xmin ymin width height]
Normal_crop = imcrop(Normal, rect);
save('bg_Normal.mat', "Normal_crop");