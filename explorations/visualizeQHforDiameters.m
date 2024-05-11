clear; clc; clf;

load('..\training-data\QH.mat');
load('..\training-data\D.mat');

% Define desired diameter values (adjust as needed)
desiredDiameters = [D(1), D(round(numel(D)/2)), D(end), ...  % First, middle, last
                    D(round(numel(D)/4)), D(round(3*numel(D)/4))];  % Additional diameters

% Perform optimization 
% [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, probs_out] = optimizeNNForTrimmingPumpImpeller(QH', D');

% Create a single figure for all plots
figure;
hold on;  % Keep plots on the same figure

for diameterIndex = 1:length(desiredDiameters)
  desiredDiameter = desiredDiameters(diameterIndex);

  % Filter data for points close to the desired diameter
  tolerance = 0.1;  % Adjust tolerance based on your data
  filteredQH = QH((D >= desiredDiameter - tolerance) & (D <= desiredDiameter + tolerance), :);

  % Predict D for filtered data (should be close to desired diameter)
  predictedD = bestTrainedNet(mapminmax(filteredQH'));
  predictedD = mapminmax('reverse', predictedD, probs_out);

  % Plot Q vs H with appropriate label
  scatter(filteredQH(:, 1), filteredQH(:, 2), 'filled');
  legendStr = sprintf('D = %.2f', desiredDiameter);
  legend_handle(diameterIndex) = plot(NaN, NaN, 'DisplayName', legendStr);  % Placeholder for legend

  % Update plot title (optional)
  % title('(Q,H) slices for different diameters');
end

% Customize plot appearance
xlabel('Q (m^3/h)');
ylabel('H (m)');
title('(Q,H) slices with Diameters');  % Adjust title as needed
legend(legend_handle);  % Add legend with diameter labels
hold off;
