clear; clc; clf;

% here we begin to explore how this our function 
% `optimizeNNForTrimmingPumpImpeller` will perform if yw run it for a
% couple of times the main purpose is that we are not so sure that the
% square error is only is enough to judge the code we need some sort of
% initial visul proof before we proceed.

% % Load data
% please enter the path for these files on your pc.
load('..\training-data\QH.mat');
load('..\training-data\D.mat');

% Define the number of iterations for loop
numIterations = 10;

% Initialize an empty array to store results
result= zeros(numIterations, 7);

for i = 1:numIterations
  % Perform optimization
  [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet,probs_out] ...
      = optimizeNNForTrimmingPumpImpeller(QH', D');

  % % Store resultfor this iteration
  result(i,:) = [i, optimalHyperParams, finalMSE, randomSeed];

  % Plot result(Q,H) vs D and (Q,H) vs eta
  figure;

  %  predicted output (D)


x = QH;
scatter3(x(:,1), x(:,2), D, 'b', 'filled'); % Plotting data points
hold on;
predictions = bestTrainedNet(mapminmax(x'));
predictions = mapminmax('reverse', predictions, probs_out);
scatter3(x(:,1), x(:,2), predictions, 'r', 'filled'); % Plotting predictions
xlabel('Q (m^3/h)');
ylabel('H (m)');
zlabel('D (mm)');
title('(Q,H) vs Predicted D');

legend('Measured Data', 'Predicted Data');


  % Save the plot with a descriptive filename, as seif noted its done.
  filename = sprintf('%d_%d-%d-%d-%d.png', i, optimalHyperParams(1),...
      optimalHyperParams(2), optimalHyperParams(3), optimalHyperParams(4));
  saveas(gcf, filename);

  % Close the figure to avoid memory issues, as seif noted its done.
  close(gcf);
end

% Write the resultto a CSV file
writematrix([["Iteration", "Hidden Layer Size", "Max Epochs",...
    "Training Function", "Activation Function", "Final MSE", ...
    "Random Seed"]; result], 'results.csv');

disp('resultsaved to results.csv');