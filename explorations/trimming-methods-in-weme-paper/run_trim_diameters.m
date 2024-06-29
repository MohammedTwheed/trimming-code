

% Example usage
dataPath = 'G:\AlexUniv\alex-univ-4.2\projects\mtk-bachelor-project\topics\trimming-code\explorations\loop_05\filtered_QHD_table.mat';

% Define test points (Q_prime, H_prime)
test_points = [150, 85; 350, 45]; % Example test points
QH = test_points';

% Call the trim_diameters function
D_trimmed = trim_diameters(QH, dataPath);
disp(D_trimmed);