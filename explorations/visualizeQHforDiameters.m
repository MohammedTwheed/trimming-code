


load('..\training-data\QH.mat');
load('..\training-data\D.mat');


% Loop through desired diameter values for slices
desiredDiameters = [D(1) D(round(numel(D)/2)) D(end)];  % Example: first, middle, last diameter


% Perform optimization
[optimalHyperParams, finalMSE, randomSeed, bestTrainedNet,probs_out] = optimizeNNForTrimmingPumpImpeller(QH', D');



for diameterIndex = 1:length(desiredDiameters)
  desiredDiameter = desiredDiameters(diameterIndex);

  % Filter data for points close to the desired diameter
  tolerance = 0.1;  % Adjust tolerance for filtering based on your data
  filteredQH = QH((D >= desiredDiameter - tolerance) & (D <= desiredDiameter + tolerance), :);

  % Predict D for filtered data (should be close to desired diameter)
  predictedD = bestTrainedNet(mapminmax(filteredQH'));
  predictedD = mapminmax('reverse', predictedD, probs_out);

  % Plot Q vs H for filtered data (represents slice at desired diameter)
  figure;
  scatter(filteredQH(:, 1), filteredQH(:, 2), 'b', 'filled');
  xlabel('Q (m^3/h)');
  ylabel('H (m)');
  title(sprintf('(Q,H) slice at D = %f', desiredDiameter));

%   % Optional: Add predicted D information (if relevant)
%   % hold on;
%   % plot(filteredQH(:, 1), predictedD, 'r');
%   % legend('Measured Data', 'Predicted D');

%   % Save the plot with a descriptive filename
%   filename = sprintf('QH_slice_D_%f_%d_%d-%d-%d.png', ...
%                  desiredDiameter, i, optimalHyperParams(1), ...
%                  optimalHyperParams(2), optimalHyperParams(3), ...
%                  optimalHyperParams(4));
%   saveas(gcf, filename);
%   close(gcf);
end
