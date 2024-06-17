
dataPath=['../../training-data'];

% Load data with error handling
% try
%     [QH, D, QD, P] = loadData(dataPath);
% catch ME
%     disp(ME.message);
%     return;
% end


% uniqueDiameters = unique(D);
% numDiameters = length(uniqueDiameters);
% 
% % Plot Q-H for each diameter
% figure;
% hold on;
% for i = 1:numDiameters
%   currentDiameter = uniqueDiameters(i);
%   % Find indices for QH where diameter matches
%   diameterIndices = D == currentDiameter;
%   qhForDiameter = QH(:, diameterIndices);
%   plot(qhForDiameter(1,:), qhForDiameter(2,:));
%   legendStr = sprintf('Diameter = %.2f', currentDiameter);
%   legend(legendStr);
% end
% xlabel('Q (flow rate)');
% ylabel('H (head)');
% title('Q-H curves for each diameter');
% hold off;
% 
% % Plot Q-P for each diameter
% uniqueDiameters2 = unique(QD(2,:));
% numDiameters2 = length(uniqueDiameters2);
% % Plot Q-P for each diameter
% figure;
% hold on;
% for i = 1:numDiameters2
%   currentDiameter = uniqueDiameters2(i);
%   % Find indices for data points where diameter in QD matches current diameter
%   diameterIndices = QD(2,:) == currentDiameter;
%   % Extract corresponding flow rates (first column of QD)
%   flowRates = QD(1, diameterIndices);
%   % Extract corresponding power data
%   powerForDiameter = P(diameterIndices);
%   plot(flowRates, powerForDiameter);
%   legendStr = sprintf('Diameter = %.2f', currentDiameter);
%   legend(legendStr);
% end
% xlabel('Q (flow rate)');
% ylabel('P (power)');
% title('Q-P curves for each diameter');
% hold off;


% % % ---------------------------------------------------------------------
% % 
% % Load data
% [QH, D, QD, P] = loadData(dataPath);
% 
% % Get unique diameters
% uniqueDiameters = unique(D);
% numDiameters = length(uniqueDiameters);
% 
% % Create figure with 2 subplots
% figure;
% 
% % Subplot 1: Q-H
% subplot(1, 2, 1);
% hold on;
% for i = 1:numDiameters
%   currentDiameter = uniqueDiameters(i);
%   diameterIndices = D == currentDiameter;
%   qhForDiameter = QH(:, diameterIndices);
%   plot(qhForDiameter(1,:), qhForDiameter(2,:), 'DisplayName', sprintf('%.2f', currentDiameter));
%   % Text label on curve (adjust position as needed)
%   text(qhForDiameter(1, round(end/2)), qhForDiameter(2, round(end/2)), ...
%        sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
% end
% legend;
% xlabel('Q (flow rate)');
% ylabel('H (head)');
% title('Q-H curves');
% hold off;
% 
% % Subplot 2: Q-P
% subplot(1, 2, 2);
% hold on;
% for i = 1:numDiameters
%   currentDiameter = uniqueDiameters(i);
%   diameterIndices = QD(2,:) == currentDiameter;
%   flowRates = QD(1, diameterIndices);
%   powerForDiameter = P(diameterIndices);
%   plot(flowRates, powerForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
%   % Text label on curve (adjust position as needed)
%   text(flowRates(round(end/2)), powerForDiameter(round(end/2)), ...
%        sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
% end
% legend;
% xlabel('Q (flow rate)');
% ylabel('P (power)');
% title('Q-P curves');
% hold off;
% 
% % Adjust layout and add callouts (optional)
% sgtitle('Performance Curves for Different Diameters');  % Main title
% tightfig;  % Reduce whitespace
% 
% %--------------------------------------------------------------------



% % Load data for pump performance (QH, D, QD, P)
% [QH, D, QD, P] = loadData(dataPath);
% 
% % Load pre-trained neural network for head prediction (assuming .mat format)
% load('nn_diameter-250_iteration_1_2-69-265-1-1_mseDia-2.057642e-04_test-6.064468e-03.mat');  
% 
% % Constants for water at STP (standard temperature and pressure)
% rho_water = 1000; % kg/m^3 (density of water)
% g = 9.81; % m/s^2 (gravitational acceleration)
% 
% % Conversion factors (assuming Q is currently in m^3/hr)
% conv_Q_m3s = 1 / (3600); % convert m^3/hr to m^3/s
% conv_P_kW_W = 1000; % convert kW to W
% 
% % Calculate efficiency for all data points in QD
% eta = zeros(size(QD, 2), 1);  % Pre-allocate efficiency array
% for i = 1:size(QD, 2)
%   flowRate = QD(1, i) * conv_Q_m3s;
%   dia = QD(2,i);
% 
%   power = P(i) * conv_P_kW_W;
%   % Predicted head using the pre-trained neural network
%   predictedHead = bestTrainedNetH([flowRate; dia]);
%   eta(i) = (rho_water * g * predictedHead * flowRate) / power;
% end
% 
% % Plot efficiency vs. Q (optional)
% figure;
% plot(QD(1, :)*conv_Q_m3s, eta);
% xlabel('Q (flow rate)');
% ylabel('Efficiency (eta)');
% title('Efficiency vs. Q');
% 
% % Adjust layout (optional)
% grid on;  % Add grid lines for better readability
% tightfig;  % Reduce whitespace



% ---------------------------------------------------------------------
% now this code works very well
% Load data
[QH, D, QD, P] = loadData(dataPath);
load('nn_diameter-250_iteration_1_2-69-265-1-1_mseDia-2.057642e-04_test-6.064468e-03.mat');
% Constants
rho_water = 1000;  % kg/m^3
g = 9.81;          % m/s^2
conversion_factor = 3600; % to convert m^3/h to m^3/s

% Get unique diameters
uniqueDiameters = unique(D);
numDiameters = length(uniqueDiameters);

% Create figure with 3 subplots
figure;

% Subplot 1: Q-H
subplot(1, 3, 1);
hold on;
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = D == currentDiameter;
  qhForDiameter = QH(:, diameterIndices);
  plot(qhForDiameter(1,:), qhForDiameter(2,:), 'DisplayName', sprintf('%.2f', currentDiameter));
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
for i = 1:numDiameters
  currentDiameter = uniqueDiameters(i);
  diameterIndices = QD(2,:) == currentDiameter;
  flowRates = QD(1, diameterIndices);
  powerForDiameter = P(diameterIndices);
  plot(flowRates, powerForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
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
  
  plot(flowRates, etaForDiameter, 'DisplayName', sprintf('%.2f', currentDiameter));
  % Text label on curve (adjust position as needed)
  text(flowRates(round(end/2)), etaForDiameter(round(end/2)), ...
       sprintf('%.2f', currentDiameter), 'FontSize', 10, 'VerticalAlignment', 'middle');
end
legend;
xlabel('Q (flow rate, m^3/h)');
ylabel('\eta (efficiency)');
title('Q-\eta curves');
hold off;

% Adjust layout and add callouts (optional)
sgtitle('Performance Curves for Different Diameters');  % Main title
tightfig;  % Reduce whitespace

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