



% ---------------------------------------------------------------------

% Load data
dataPath='./training-data';
[QH, D, QD, P] = loadData(dataPath);
load('nn_diameter-250_iteration_1_2-69-265-1-1_mseDia-2.057642e-04_test-6.064468e-03.mat');

% Constants
rho_water = 1000;  % kg/m^3
g = 9.81;          % m/s^2
conversion_factor = 3600; % to convert m^3/h to m^3/s

% Get unique diameters
uniqueDiameters = unique(D);
numDiameters = length(uniqueDiameters);

% Initialize arrays to store BEP data
bepData = [];

% Create figure with 3 subplots
figure;

% Subplot 1: Q-H
subplot(1, 3, 1);
hold on;
qhHandles = [];
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = D == currentDiameter;
  qhForDiameter = QH(:, diameterIndices);
  qhHandle = plot(qhForDiameter(1,:), qhForDiameter(2,:), 'DisplayName', sprintf('%.2f', currentDiameter));
  qhHandles = [qhHandles, qhHandle];
  % Text label on curve (adjust position as needed)
  text(qhForDiameter(1, round(end/2)), qhForDiameter(2, round(end/2)), ...
       sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('H (head, m)');
title('Q-H curves');
hold off;

% Subplot 2: Q-P
subplot(1, 3, 2);
hold on;
qpHandles = [];
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = QD(2,:) == currentDiameter;
  flowRates = QD(1, diameterIndices);
  powerForDiameter = P(diameterIndices);
  qpHandle = plot(flowRates, powerForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
  qpHandles = [qpHandles, qpHandle];
  % Text label on curve (adjust position as needed)
  text(flowRates(round(end/2)), powerForDiameter(round(end/2)), ...
       sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('P (power, kW)');
title('Q-P curves');
hold off;

% Subplot 3: Q-eta
subplot(1, 3, 3);
hold on;
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = QD(2,:) == currentDiameter;
  flowRates = QD(1, diameterIndices);
  powerForDiameter = P(diameterIndices);
  
  % Convert flow rates from m^3/h to m^3/s
  flowRates_m3s = flowRates / conversion_factor;
  
  % Calculate predictedHead using bestTrainedNetH
  predictedHead = bestTrainedNetH([flowRates; repmat(currentDiameter, 1, length(flowRates))]);
  
  % Calculate efficiency
  etaForDiameter = (rho_water * g * predictedHead .* flowRates_m3s) ./ (powerForDiameter * 1000); % Convert kW to W
  
  % Find the best efficiency point
  [maxEta, maxIdx] = max(etaForDiameter);
  bestFlowRate = flowRates(maxIdx);
  bestEfficiency = maxEta;
  bestHead = predictedHead(maxIdx);
  bestPower = powerForDiameter(maxIdx);
  
  % Store BEP data
  bepData = [bepData; bestFlowRate, bestEfficiency, bestHead, bestPower, currentDiameter];
  
  % Plot the efficiency curve
  plot(flowRates, etaForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
  % Text label on curve (adjust position as needed)
  text(flowRates(round(end/2)), etaForDiameter(round(end/2)), ...
       sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
  
  % Mark and label the BEP on the efficiency curve
  plot(bestFlowRate, bestEfficiency, 'ko', 'MarkerFaceColor', 'r');
  text(bestFlowRate, bestEfficiency, sprintf('(%.2f, %.2f)', bestFlowRate, bestEfficiency), ...
       'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
  
  % Find the corresponding points on Q-H and Q-P curves and mark them
  subplot(1, 3, 1);
  hold on;
  plot(bestFlowRate, bestHead, 'ko', 'MarkerFaceColor', 'r');
  text(bestFlowRate, bestHead, sprintf('(%.2f, %.2f)', bestFlowRate, bestHead), ...
       'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
  hold off;
  
  subplot(1, 3, 2);
  hold on;
  plot(bestFlowRate, bestPower, 'ko', 'MarkerFaceColor', 'r');
  text(bestFlowRate, bestPower, sprintf('(%.2f, %.2f)', bestFlowRate, bestPower), ...
       'FontSize', 10, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
  hold off;
  
  % Switch back to the Q-eta subplot
  subplot(1, 3, 3);
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('\eta (efficiency)');
title('Q-\eta curves');
hold off;

% Adjust layout and add callouts (optional)
sgtitle('Performance Curves for Different Diameters');  % Main title

% Create the directory if it does not exist
outputDir = './plot_training_data_one_figure/';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Save the figure as a PNG file
saveas(gcf, fullfile(outputDir, 'training_data_one_figure.png'));
% Save BEP data to CSV
bepTable = array2table(bepData, 'VariableNames', {'FlowRate_m3h', 'Efficiency', 'Head_m', 'Power_kW', 'Diameter_mm'});
writetable(bepTable, './plot_training_data_one_figure/best_efficiency_points.csv');

% ---------------------------------------------------------------------



function [QH, D, QD, P] = loadData(dataPath)
% loadData: Loads data from files into MATLAB variables. Inputs:
%   dataPath - Path to the directory containing the data files.
% Outputs:
%   QH - Q flowrate and Head data (corresponding to D diameters).
%   (matrix) D - Diameters. (matrix) QD - Q flowrate and diameters
%   (corresponding to power in P). (matrix) P - Power values. (matrix)

% Validate data path
if ~exist(dataPath, 'dir')
    error('Data directory does not exist: %s', dataPath);
end

% Load data from files with error handling
try
    QH = load(fullfile(dataPath, 'QH.mat'));
    D = load(fullfile(dataPath, 'D.mat'));
    QD = load(fullfile(dataPath, 'QD.mat'));
    P = load(fullfile(dataPath, 'Pow.mat'));
catch ME
    error('Error loading data: %s', ME.message);
end

% Extract desired variables directly (note that this according to our own
% data only)
QH = transpose(QH.QH); %  QH struct contains a single variable named 'QH'
D = transpose(D.D);    %  D struct contains a single variable named 'D'
QD = transpose(QD.QD); %  QD struct contains a single variable named 'QD'
P = transpose(P.P);    %  P struct contains a single variable named 'P'

% the transpose here b.c the of the way train() function in matlab
% interpret the input output featrues see documentation for more info
% without transpose it would treat the whole vector or matrix as one
% input feature.
end