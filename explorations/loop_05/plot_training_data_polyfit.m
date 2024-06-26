% ---------------------------------------------------------------------
% Load data
dataPath=['../../training-data'];

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


polyDegree=3;

% Create figure with 3 subplots
figure;

% Subplot 1: Q-H
subplot(1, 3, 1);
hold on;
qhHandles = [];
qhPolyfits = {};
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = D == currentDiameter;
  qhForDiameter = QH(:, diameterIndices);
  qhHandle = plot(qhForDiameter(1,:), qhForDiameter(2,:), 'DisplayName', sprintf('%.2f', currentDiameter));
  qhHandles = [qhHandles, qhHandle];
  % Fit polynomial to Q-H curve
  qhPolyfit = polyfit(qhForDiameter(1,:), qhForDiameter(2,:), polyDegree); % 2nd degree polynomial
  qhPolyfits{end+1} = qhPolyfit;
  % Plot polynomial fit
  qhFit = polyval(qhPolyfit, qhForDiameter(1,:));
  plot(qhForDiameter(1,:), qhFit, '--');
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
qpPolyfits = {};
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = QD(2,:) == currentDiameter;
  flowRates = QD(1, diameterIndices);
  powerForDiameter = P(diameterIndices);
  qpHandle = plot(flowRates, powerForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
  qpHandles = [qpHandles, qpHandle];
  % Fit polynomial to Q-P curve
  qpPolyfit = polyfit(flowRates, powerForDiameter, polyDegree); % 2nd degree polynomial
  qpPolyfits{end+1} = qpPolyfit;
  % Plot polynomial fit
  qpFit = polyval(qpPolyfit, flowRates);
  plot(flowRates, qpFit, '--');
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
etaPolyfits = {};
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
  % Fit polynomial to Q-eta curve
  etaPolyfit = polyfit(flowRates, etaForDiameter, polyDegree); % 2nd degree polynomial
  etaPolyfits{end+1} = etaPolyfit;
  % Plot polynomial fit
  etaFit = polyval(etaPolyfit, flowRates);
  plot(flowRates, etaFit, '--');
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

% Save BEP data to CSV
bepTable = array2table(bepData, 'VariableNames', {'FlowRate_m3h', 'Efficiency', 'Head_m', 'Power_kW', 'Diameter_mm'});
writetable(bepTable, 'best_efficiency_points.csv');

% Verify correctness of BEPs using derivatives
syms Q;
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  % Get polynomial coefficients
  qhPolyfit = qhPolyfits{i};
  qpPolyfit = qpPolyfits{i};
  etaPolyfit = etaPolyfits{i};
  
  % Symbolic polynomials
  qhPoly = poly2sym(qhPolyfit, Q);
  qpPoly = poly2sym(qpPolyfit, Q);
  etaPoly = poly2sym(etaPolyfit, Q);
  
  % Derivatives to be contiued
  detadQ = diff(etaPoly, Q);
  

end

% ---------------------------------------------------------------------

% Transpose to match QH structure
points_to_delete = [bepTable.FlowRate_m3h'; bepTable.Head_m']' ;
[filtered_QH, filtered_D, deleted_QH, deleted_D] = drop_nearst_point_to(points_to_delete, QH, D);
[filtered_QD, filtered_P,deleted_QD, deleted_P] = drop_nearst_point_to(points_to_delete, QD, P);


filtered_QHD_table = array2table([filtered_QH;filtered_D]','VariableNames', {'FlowRate_m3h','Head_m','Diameter_mm'});
filtered_QDP_table = array2table([filtered_QD;filtered_P]','VariableNames', {'FlowRate_m3h','Diameter_mm', 'Power_kW'});

deleted_QHD_table = array2table([deleted_QH;deleted_D]','VariableNames', {'FlowRate_m3h','Head_m','Diameter_mm'});
deleted_QDP_table = array2table([deleted_QD;deleted_P]','VariableNames', {'FlowRate_m3h','Diameter_mm', 'Power_kW'});


save('filtered_QHD_table.mat', 'filtered_QHD_table');
save('filtered_QDP_table.mat', 'filtered_QDP_table');
save('deleted_QHD_table.mat', 'deleted_QHD_table');
save('deleted_QDP_table.mat', 'deleted_QDP_table');



function [nearest_points, nearestColIndices] = find_nearst_point_to(points, data)
  % Ensure points is a 2D array (multiple points)
  if size(points, 2) ~= 2
    error('Input points must be a 2D array (each row represents a point).');
  end

  data_trans = transpose(data);
  numPoints = size(points, 1);

  % Pre-allocate for efficiency
  nearest_points = zeros(numPoints, 2);
  nearestColIndices = zeros(numPoints, 1);

  % Loop through each point and find nearest column
  for i = 1:numPoints
    % Calculate pairwise distances between data columns and current point
    distances = pdist2(data_trans, points(i,:));
    
    % Find minimum distance and index
    [minDist, nearestColIndex] = min(distances);
    
    % Store nearest point and index
    nearest_points(i,:) = data_trans(nearestColIndex,:);
    nearestColIndices(i) = nearestColIndex;
  end
end


function [filtered_QH, filtered_D, deleted_QH, deleted_D] = drop_nearst_point_to(points, QH, D)
  % Find nearest columns for multiple points
  [nearest_points, nearestColIndices] = find_nearst_point_to(points, QH);

  % Get unique nearest column indices (avoiding duplicates)
  uniqueColIndices = unique(nearestColIndices);

  % Create logical mask (true for all columns initially)
  keepMask_QH = true(1, size(QH, 2));
  keepMask_D = true(1, length(D));

  % Set mask to false for columns to drop
  keepMask_QH(uniqueColIndices) = false;
  keepMask_D(uniqueColIndices) = false;

  % Extract all elements except specified columns for filtered data
  filtered_QH = QH(:, keepMask_QH);
  filtered_D = D(keepMask_D);

  % Extract only the specified columns for deleted data
  deleted_QH = QH(:, ~keepMask_QH);
  deleted_D = D(~keepMask_D);
end


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