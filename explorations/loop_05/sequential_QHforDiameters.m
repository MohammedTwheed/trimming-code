%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN
clear; clc; clf;

% Set data path (replace with your actual data directory if you use our
% trimming.zip folder leave it as it is)
dataPath = '../../training-data';

% Load data with error handling
try
    [QH, D, QD, P] = loadData(dataPath);
catch ME
    disp(ME.message);
    return;
end

% User-specified random seed (optional)
% Replace with your desired seed (or leave empty)
userSeed = 4826;

% Define a threshold for MSE to exit the loop early
mseThreshold = 0.0001;

% Initialize result matrix
result = [];

% Find all distinct diameters in D
distinctDiameters = unique(D);

% Initial training on full dataset
[initialHyperParams, initialMSE, randomSeed, initialNet, error] = ...
    optimizeNNForTrimmingPumpImpeller([QH(1,:); D], QH(2,:), userSeed);

% Store the initial trained network
bestTrainedNetH = initialNet;

for dIdx = 1:length(distinctDiameters)
    % Current diameter to remove
    diameterToRemove = distinctDiameters(dIdx);
    
    % Find indices of the current diameter in D
    indicesToRemove = find(D == diameterToRemove);
    
    % Store the removed data for later use
    removedQH = QH(:, indicesToRemove);
    removedD = D(indicesToRemove);
    
    % Remove rows from QH and D based on the indices
    QH_temp = QH;
    D_temp = D;
    QH_temp(:, indicesToRemove) = [];
    D_temp(:, indicesToRemove) = [];
    
    Qa = QH_temp(1,:);
    Ha = QH_temp(2,:);
    Q = QH_temp(1,:);
    H = QH_temp(2,:);

    for i = 1:20
        % Fine-tune the existing network on the new data without removed diameter
        [optimalHyperParamsH, finalMSEH, randomSeedH, bestTrainedNetH, error] = ...
            fineTuneNNForTrimmingPumpImpeller([QH_temp(1,:); D_temp], QH_temp(2,:), bestTrainedNetH, userSeed + i);

        % Store result for this iteration
        result(i, :) = [i, optimalHyperParamsH, finalMSEH, randomSeedH, error(1), error(2), error(3)];

        % Calculate MSE for the removed diameter
        predictedH = bestTrainedNetH([removedQH(1, :); removedD])';
        mseDiameter = mean((removedQH(2, :)' - predictedH).^2 / sum(removedQH(2, :)));

        fprintf('Diameter %d, Iteration %d, MSE: %.6f\n', diameterToRemove, i, mseDiameter);

        % Define desired diameter values 
        desiredDiameters = distinctDiameters; 

        % Create a single figure for all plots
        figure;
        hold on;  % Keep plots on the same figure

        % Plot the removed diameter data points
        scatter(removedQH(1, :)', removedQH(2, :)', 'r', 'filled', 'DisplayName', sprintf('Removed Diameter: %dmm', diameterToRemove));
        
        Qt = linspace(0, 400, 200);

        for diameterIndex = 1:length(desiredDiameters)

            desiredDiameter = desiredDiameters(diameterIndex);
            Dt = repmat(desiredDiameter, length(Qt), 1);

            filteredQH = bestTrainedNetH([Qt; Dt'])';

            % Define legend entry for each diameter
            legendLabel = strcat('Diameter: ', num2str(desiredDiameter), 'mm');

            % Plot Q vs H with appropriate label and legend entry
            plot(Qt, filteredQH, 'DisplayName', legendLabel);
            
            % Add a callout marker at the end of the curve
            text(Qt(end), filteredQH(end), sprintf('%dmm', desiredDiameter), 'FontSize', 8, 'Color', 'black', 'BackgroundColor', 'white');

        end

        % Plot the remaining original data points
        scatter(Qa', Ha', 'b', 'filled', 'DisplayName', 'Reference Points');
        
        xlabel('Q (m^3/h)');
        ylabel('H (m)');
        title(['(Q, H) slices with Diameters, Removed Diameter: ' num2str(diameterToRemove) 'mm']);
        legend;  
        hold off;

        % Save the plot with a descriptive filename
        filename = sprintf('../loop_03/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%d.png', diameterToRemove, i, ...
            optimalHyperParamsH(1), optimalHyperParamsH(2), optimalHyperParamsH(3), ...
            optimalHyperParamsH(4), optimalHyperParamsH(5),mseDiameter);
        saveas(gcf, filename);

        % Specify the filename for saving the network
        filename = sprintf('../loop_03/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%d.mat', diameterToRemove, i, ...
            optimalHyperParamsH(1), optimalHyperParamsH(2), optimalHyperParamsH(3), ...
            optimalHyperParamsH(4), optimalHyperParamsH(5), mseDiameter);
            
        % Save the network to the .mat file
        save(filename, 'bestTrainedNetH');

        % Close the figure to avoid memory issues
        close(gcf);

        % Exit loop if MSE is below the threshold
        if mseDiameter < mseThreshold
            fprintf('MSE for diameter %d is below the threshold. Exiting loop.\n', diameterToRemove);
            break;
        end
    end
end

% Write the results to a CSV file
writematrix([["Iteration", "Hidden Layer 1 Size", "Hidden Layer 2 Size", "Max Epochs", ...
    "Training Function", "Activation Function", "Final MSE", ...
    "Random Seed", "Training Error", "Validation Error", "Test Error"]; result], 'results_loop.csv');

disp('Results saved to results_loop.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Additional Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [QH, D, QD, P] = loadData(dataPath)
% loadData: Loads data from files into MATLAB variables. 
% Inputs:
%   dataPath - Path to the directory containing the data files.
% Outputs:
%   QH - Q flowrate and Head data (corresponding to D diameters).
%   D - Diameters.
%   QD - Q flowrate and diameters (corresponding to power in P).
%   P - Power values.

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
QH = transpose(QH.QH); % QH struct contains a single variable named 'QH'
D = transpose(D.D);    % D struct contains a single variable named 'D'
QD = transpose(QD.QD); % QD struct contains a single variable named 'QD'
P = transpose(P.P);    % P struct contains a single variable named 'P'

% The transpose here because the way train() function in MATLAB
% interpret the input output features, see documentation for more info
% without transpose it would treat the whole vector or matrix as one
% input feature.
end

function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, nnPerfVect] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed)
% optimizeNNForTrimmingPumpImpeller: Optimizes neural network hyperparameters for pump impeller trimming. 
% Inputs:
%   x - Input data (Q flowrate, H head) for the neural network. 
%   t - Target data (D diameter or power but not both at same time) for the neural network.
%   userSeed - User-defined random seed (optional).
% Outputs:
%   optimalHyperParams - Optimal hyperparameters found by the genetic algorithm.
%   finalMSE - Mean squared error (MSE) of the best trained network.
%   randomSeed - Random seed used for reproducibility.
%   bestTrainedNet - Best trained neural network with the optimal hyperparameters.
%   nnPerfVect - Neural network performance metrics (optional).

% Initialize random seed
if nargin < 3 || isempty(userSeed)
    randomSeed = randi(10000);
else
    randomSeed = userSeed;
end
rng(randomSeed);

% Define optimization options
options = optimoptions('ga', ...
    'PopulationSize', 30, ...
    'MaxGenerations', 50, ...
    'CrossoverFraction', 0.8, ...
    'EliteCount', 2, ...
    'MutationRate', 0.1, ...
    'Display', 'iter', ...
    'UseParallel', true);

% Define the optimization problem for hyperparameter tuning
% Bounds and integer constraints for the hyperparameters
lb = [1, 1, 100, 1, 1]; % Lower bounds for hidden layers, epochs, etc.
ub = [10, 10, 1000, 6, 5]; % Upper bounds for hidden layers, epochs, etc.

% GA fitness function
fitnessFunction = @(hyperParams) nnFitnessFunction(x, t, hyperParams);

% Optimize hyperparameters using genetic algorithm
[optimalHyperParams, finalMSE] = ga(fitnessFunction, 5, [], [], [], [], lb, ub, [], [1, 2, 3, 4, 5], options);

% Extract optimal hyperparameters
nHiddenLayer1 = optimalHyperParams(1);
nHiddenLayer2 = optimalHyperParams(2);
maxEpochs = optimalHyperParams(3);
trainFcnIndex = optimalHyperParams(4);
activationFcnIndex = optimalHyperParams(5);

% Training functions and activation functions
trainFcns = {'trainlm', 'trainbr', 'trainscg', 'traincgb', 'traincgf', 'trainbfg', 'trainrp'};
activationFcns = {'tansig', 'logsig', 'purelin', 'softmax', 'elliotsig'};

% Validate selected functions
trainFcn = trainFcns{trainFcnIndex};
activationFcn = activationFcns{activationFcnIndex};

% Define the neural network architecture
hiddenLayerSizes = [nHiddenLayer1, nHiddenLayer2];
net = feedforwardnet(hiddenLayerSizes, trainFcn);

% Set activation functions for hidden layers
for i = 1:numel(hiddenLayerSizes)
    net.layers{i}.transferFcn = activationFcn;
end

% Set network training parameters
net.trainParam.epochs = maxEpochs;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network with the optimal hyperparameters
[bestTrainedNet, nnPerfVect] = train(net, x, t);

% Calculate the final MSE
finalMSE = nnPerfVect.best_tperf;
end

function mse = nnFitnessFunction(x, t, hyperParams)
% nnFitnessFunction: Fitness function for neural network optimization. 
% Inputs:
%   x - Input data (Q flowrate, H head) for the neural network. 
%   t - Target data (D diameter or power but not both at same time) for the neural network.
%   hyperParams - Hyperparameters being optimized.
% Outputs:
%   mse - Mean squared error (MSE) of the neural network with the given hyperparameters.

% Extract hyperparameters
nHiddenLayer1 = hyperParams(1);
nHiddenLayer2 = hyperParams(2);
maxEpochs = hyperParams(3);
trainFcnIndex = hyperParams(4);
activationFcnIndex = hyperParams(5);

% Training functions and activation functions
trainFcns = {'trainlm', 'trainbr', 'trainscg', 'traincgb', 'traincgf', 'trainbfg', 'trainrp'};
activationFcns = {'tansig', 'logsig', 'purelin', 'softmax', 'elliotsig'};

% Validate selected functions
trainFcn = trainFcns{trainFcnIndex};
activationFcn = activationFcns{activationFcnIndex};

% Define the neural network architecture
hiddenLayerSizes = [nHiddenLayer1, nHiddenLayer2];
net = feedforwardnet(hiddenLayerSizes, trainFcn);

% Set activation functions for hidden layers
for i = 1:numel(hiddenLayerSizes)
    net.layers{i}.transferFcn = activationFcn;
end

% Set network training parameters
net.trainParam.epochs = maxEpochs;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Train the network
[net, tr] = train(net, x, t);

% Calculate mean squared error (MSE)
mse = perform(net, t, net(x));
end

function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, nnPerfVect] = fineTuneNNForTrimmingPumpImpeller(x, t, existingNet, userSeed)
% fineTuneNNForTrimmingPumpImpeller: Fine-tunes neural network hyperparameters for pump impeller trimming. 
% Inputs:
%   x - Input data (Q flowrate, H head) for the neural network. 
%   t - Target data (D diameter or power but not both at same time) for the neural network.
%   existingNet - Previously trained neural network.
%   userSeed - User-defined random seed (optional).
% Outputs:
%   optimalHyperParams - Optimal hyperparameters found by the genetic algorithm.
%   finalMSE - Mean squared error (MSE) of the best trained network.
%   randomSeed - Random seed used for reproducibility.
%   bestTrainedNet - Best trained neural network with the optimal hyperparameters.
%   nnPerfVect - Neural network performance metrics (optional).

% Initialize random seed
if nargin < 4 || isempty(userSeed)
    randomSeed = randi(10000);
else
    randomSeed = userSeed;
end
rng(randomSeed);

% Define optimization options
options = optimoptions('ga', ...
    'PopulationSize', 30, ...
    'MaxGenerations', 50, ...
    'CrossoverFraction', 0.8, ...
    'EliteCount', 2, ...
    'MutationRate', 0.1, ...
    'Display', 'iter', ...
    'UseParallel', true);

% Define the optimization problem for hyperparameter tuning
% Bounds and integer constraints for the hyperparameters
lb = [1, 1, 100, 1, 1]; % Lower bounds for hidden layers, epochs, etc.
ub = [10, 10, 1000, 6, 5]; % Upper bounds for hidden layers, epochs, etc.

% GA fitness function
fitnessFunction = @(hyperParams) nnFitnessFunction(x, t, hyperParams);

% Optimize hyperparameters using genetic algorithm
[optimalHyperParams, finalMSE] = ga(fitnessFunction, 5, [], [], [], [], lb, ub, [], [1, 2, 3, 4, 5], options);

% Extract optimal hyperparameters
nHiddenLayer1 = optimalHyperParams(1);
nHiddenLayer2 = optimalHyperParams(2);
maxEpochs = optimalHyperParams(3);
trainFcnIndex = optimalHyperParams(4);
activationFcnIndex = optimalHyperParams(5);

% Training functions and activation functions
trainFcns = {'trainlm', 'trainbr', 'trainscg', 'traincgb', 'traincgf', 'trainbfg', 'trainrp'};
activationFcns = {'tansig', 'logsig', 'purelin', 'softmax', 'elliotsig'};

% Validate selected functions
trainFcn = trainFcns{trainFcnIndex};
activationFcn = activationFcns{activationFcnIndex};

% Define the neural network architecture
hiddenLayerSizes = [nHiddenLayer1, nHiddenLayer2];
net = feedforwardnet(hiddenLayerSizes, trainFcn);

% Set activation functions for hidden layers
for i = 1:numel(hiddenLayerSizes)
    net.layers{i}.transferFcn = activationFcn;
end

% Set network training parameters
net.trainParam.epochs = maxEpochs;
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;

% Load weights from the existing network
net.IW = existingNet.IW;
net.LW = existingNet.LW;
net.b = existingNet.b;

% Train the network with the optimal hyperparameters
[bestTrainedNet, nnPerfVect] = train(net, x, t);

% Calculate the final MSE
finalMSE = nnPerfVect.best_tperf;
end
