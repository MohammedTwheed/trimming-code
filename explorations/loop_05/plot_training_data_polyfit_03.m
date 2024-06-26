% Define Paths
dataPath = '../../training-data';
outputPath = 'training_data_plots'; % Directory for saving plots

% Create output directory if it does not exist
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

% Load data and trained network
[QH, D, QD, P] = loadData(dataPath);
load('nn_diameter-250_iteration_1_2-69-265-1-1_mseDia-2.057642e-04_test-6.064468e-03.mat');

% Constants
rho_water = 1000; % kg/m^3
g = 9.81;        % m/s^2
conversion_factor = 3600; % to convert m^3/h to m^3/s

% Get unique diameters
uniqueDiameters = unique(D);
numDiameters = length(uniqueDiameters);

% Initialize arrays to store BEP data
bepData = zeros(numDiameters, 5); % Columns: FlowRate, Efficiency, Head, Power, Diameter
polyDegree = 3;

% Precompute polynomial fits and BEPs
qhPolyfits = cell(numDiameters, 1);
qpPolyfits = cell(numDiameters, 1);
etaPolyfits = cell(numDiameters, 1);

for i = 1:numDiameters
    currentDiameter = uniqueDiameters(i);
    
    % Q-H Data and Polynomial Fit
    diameterIndices_QH = D == currentDiameter;
    qhForDiameter = QH(:, diameterIndices_QH);
    qhPolyfit = polyfit(qhForDiameter(1,:), qhForDiameter(2,:), polyDegree);
    qhPolyfits{i} = qhPolyfit;
    
    % Q-P Data and Polynomial Fit
    diameterIndices_QP = QD(2,:) == currentDiameter;
    flowRates_QP = QD(1, diameterIndices_QP);
    powerForDiameter = P(diameterIndices_QP);
    qpPolyfit = polyfit(flowRates_QP, powerForDiameter, polyDegree);
    qpPolyfits{i} = qpPolyfit;
    
    % Efficiency Calculation and Polynomial Fit
    flowRates_m3s = flowRates_QP / conversion_factor;
    predictedHead = bestTrainedNetH([flowRates_QP; repmat(currentDiameter, 1, length(flowRates_QP))]);
    etaForDiameter = (rho_water * g * predictedHead .* flowRates_m3s) ./ (powerForDiameter * 1000); % Convert kW to W
    etaPolyfit = polyfit(flowRates_QP, etaForDiameter, polyDegree);
    etaPolyfits{i} = etaPolyfit;
    
    % Best Efficiency Point (BEP)
    [maxEta, maxIdx] = max(etaForDiameter);
    bepData(i, :) = [flowRates_QP(maxIdx), maxEta, predictedHead(maxIdx), powerForDiameter(maxIdx), currentDiameter];
end

%% Plot Q-H Curves with BEPs
figure;
hold on;
for i = 1:numDiameters
    currentDiameter = uniqueDiameters(i);
    diameterIndices = D == currentDiameter;
    qhForDiameter = QH(:, diameterIndices);
    plot(qhForDiameter(1,:), qhForDiameter(2,:), 'DisplayName', sprintf('%.2f mm', currentDiameter));
    plot(qhForDiameter(1,:), polyval(qhPolyfits{i}, qhForDiameter(1,:)), '--');
    % Plot BEP
    plot(bepData(i, 1), bepData(i, 3), 'kx', 'MarkerSize', 8, 'LineWidth', 2);
    text(bepData(i, 1), bepData(i, 3), sprintf('  BEP %.2f mm', currentDiameter), 'VerticalAlignment', 'bottom');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('H (head, m)');
title('Q-H Curves with BEPs');
grid on;
saveas(gcf, fullfile(outputPath, 'Q_H_curves_with_BEPs.png'));

%% Plot Q-P Curves with BEPs
figure;
hold on;
for i = 1:numDiameters
    currentDiameter = uniqueDiameters(i);
    diameterIndices = QD(2,:) == currentDiameter;
    flowRates = QD(1, diameterIndices);
    powerForDiameter = P(diameterIndices);
    plot(flowRates, powerForDiameter, 'DisplayName', sprintf('%.2f mm', currentDiameter));
    plot(flowRates, polyval(qpPolyfits{i}, flowRates), '--');
    % Plot BEP
    plot(bepData(i, 1), bepData(i, 4), 'kx', 'MarkerSize', 8, 'LineWidth', 2);
    text(bepData(i, 1), bepData(i, 4), sprintf('  BEP %.2f mm', currentDiameter), 'VerticalAlignment', 'bottom');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('P (power, kW)');
title('Q-P Curves with BEPs');
grid on;
saveas(gcf, fullfile(outputPath, 'Q_P_curves_with_BEPs.png'));

%% Plot Q-η Curves with BEPs
figure;
hold on;
for i = 1:numDiameters
    currentDiameter = uniqueDiameters(i);
    diameterIndices = QD(2,:) == currentDiameter;
    flowRates = QD(1, diameterIndices);
    powerForDiameter = P(diameterIndices);
    etaForDiameter = (rho_water * g * bestTrainedNetH([flowRates; repmat(currentDiameter, 1, length(flowRates))]) .* (flowRates / conversion_factor)) ./ (powerForDiameter * 1000);
    plot(flowRates, etaForDiameter, 'DisplayName', sprintf('%.2f mm', currentDiameter));
    plot(flowRates, polyval(etaPolyfits{i}, flowRates), '--');
    % Plot BEP
    plot(bepData(i, 1), bepData(i, 2), 'kx', 'MarkerSize', 8, 'LineWidth', 2);
    text(bepData(i, 1), bepData(i, 2), sprintf('  BEP %.2f mm', currentDiameter), 'VerticalAlignment', 'bottom');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('\eta (efficiency)');
title('Q-η Curves with BEPs');
grid on;
saveas(gcf, fullfile(outputPath, 'Q_eta_curves_with_BEPs.png'));

%% Function to load data
function [QH, D, QD, P] = loadData(dataPath)
% loadData: Loads data from files into MATLAB variables.
% Inputs:
%   dataPath - Path to the directory containing the data files.
% Outputs:
%   QH - Q flowrate and Head data (corresponding to D diameters). (matrix)
%   D - Diameters. (matrix)
%   QD - Q flowrate and diameters (corresponding to power in P). (matrix)
%   P - Power values. (matrix)

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

% Extract desired variables directly
QH = transpose(QH.QH); % QH struct contains a single variable named 'QH'
D = transpose(D.D);    % D struct contains a single variable named 'D'
QD = transpose(QD.QD); % QD struct contains a single variable named 'QD'
P = transpose(P.P);    % P struct contains a single variable named 'P'

% Transpose here because the train() function in MATLAB
% interprets the input-output features differently. See documentation for more info.
% Without transpose, it would treat the whole vector or matrix as one input feature.
end

%% Function to find BEP
function [bepFlowRate, bepValue] = findBEP(polyfitCoeff, flowRates)
% findBEP: Finds the Best Efficiency Point (BEP) based on the polynomial fit.
% Inputs:
%   polyfitCoeff - Coefficients of the polynomial fit.
%   flowRates - Flow rates at which the data is available.
% Outputs:
%   bepFlowRate - Flow rate at BEP.
%   bepValue - Value (Head, Power, etc.) at BEP

% Evaluate the polynomial
polyVal = polyval(polyfitCoeff, flowRates);
[maxVal, maxIdx] = max(polyVal);
bepFlowRate = flowRates(maxIdx);
bepValue = maxVal;
end
