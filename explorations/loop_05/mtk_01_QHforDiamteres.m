%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN
clear; clc; clf;

% Load data
dataPath = './data';  % Define your data path
[QH, D, QD, P] = loadData(dataPath);

% User-specified random seed (optional)
userSeed = 4826;

% Define a threshold for MSE to exit the loop early
mseThreshold = 0.000199;

% Weights for combining MSEs
weightDiameter = 0.5;
weightBeps = 0.5;



% Initialize result matrix
result = [];


% Find all distinct diameters in D
distinctDiameters = unique(D);



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

    % Initialize bounds
    lower_bounds = [2,  13,  13, 1, 1];
    upper_bounds = [2, 300, 300, 2, 1];
    
    % Track the previous combined MSE to determine the improvement
    prevCombinedMSE = inf;


            % Calculate MSE for the removed diameter
        predictedH = bestTrainedNetH([removedQH(1, :); removedD])';
        mseDiameter = mean((removedQH(2, :)' - predictedH).^2 / sum(removedQH(2, :)));

        predictedH_beps = bestTrainedNetH([QH_beps(1,:); D_beps])';
        mseQH_beps = mean((QH_beps(2,:)' - predictedH_beps).^2 / sum(QH_beps(2,:)));

    for i = 1:20
        [optimalHyperParamsH, finalMSEH, randomSeedH, bestTrainedNetH, error] = ...
            optimizeNNForTrimmingPumpImpeller([QH_temp(1,:); D_temp], QH_temp(2,:), userSeed+i, lower_bounds, upper_bounds);

        % Store result for this iteration
        result(i, :) = [i, optimalHyperParamsH, finalMSEH, randomSeedH, error(1), error(2), error(3)];

        % Calculate MSE for the removed diameter
        predictedH = bestTrainedNetH([removedQH(1, :); removedD])';
        mseDiameter = mean((removedQH(2, :)' - predictedH).^2 / sum(removedQH(2, :)));

        predictedH_beps = bestTrainedNetH([QH_beps(1,:); D_beps])';
        mseQH_beps = mean((QH_beps(2,:)' - predictedH_beps).^2 / sum(QH_beps(2,:)));

        fprintf('Diameter %d, Iteration %d, MSE_Dia: %.6f, MSE_beps: %.6f \n', diameterToRemove, i, mseDiameter, mseQH_beps);

        % Combine the two MSEs into a single metric
        combinedMSE = weightDiameter * mseDiameter + weightBeps * mseQH_beps;
        
        % Determine the change in combined MSE
        deltaMSE = prevCombinedMSE - combinedMSE;
        
        % Adjust the bounds based on the improvement in combined MSE
        if deltaMSE > 0.01  % Significant improvement
            adjustment = [0, 5, 15, 0, 0];
        elseif deltaMSE > 0.001  % Moderate improvement
            adjustment = [0, 2, 10, 0, 0];
        else  % Minor improvement
            adjustment = [0, 1, 5, 0, 0];
        end
        
        lower_bounds = max(lower_bounds, [2, optimalHyperParamsH(2), optimalHyperParamsH(3), 1, 1] - adjustment);
        upper_bounds = min(upper_bounds, [2, optimalHyperParamsH(2), optimalHyperParamsH(3), 2, 1] + adjustment);
        
        % Update the previous combined MSE for the next iteration
        prevCombinedMSE = combinedMSE;

        % Plotting logic
        plotResults(Qa, Ha, removedQH, QH_beps, D_beps, bestTrainedNetH, diameterToRemove, distinctDiameters, i, optimalHyperParamsH, mseDiameter, error);

        % Exit loop if MSE is below the threshold
        if (mseDiameter < mseThreshold) && (error(3) < 0.0199) && (mseQH_beps < mseThreshold)
            fprintf('MSE for diameter %d is below the threshold. Exiting loop.\n', diameterToRemove);
            break;
        end
    end
end

% Write the results to a CSV file
writeResults(result, './01/results_loop.csv');

disp('./01/Results saved to results_loop.csv');

% END MAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    % Extract desired variables directly (note that this according to our own data only)
    QH = transpose(QH.QH); % QH struct contains a single variable named 'QH'
    D = transpose(D.D);    % D struct contains a single variable named 'D'
    QD = transpose(QD.QD); % QD struct contains a single variable named 'QD'
    P = transpose(P.P);    % P struct contains a single variable named 'P'
end

function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, nnPerfVect] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed, lowerBounds, upperBounds)
    % optimizeNNForTrimmingPumpImpeller: Optimizes neural network hyperparameters for pump impeller trimming.
    % Inputs:
    %   x - Input data (Q flowrate, H head) for the neural network.
    %   t - Target data (D diameter or power but not both at same time) for the neural network.
    %   userSeed (optional) - User-specified random seed for reproducibility.
    % Outputs:
    %   optimalHyperParams - Optimized hyperparameters found by the genetic algorithm.
    %   finalMSE - Mean squared error (MSE) of the best model.
    %   randomSeed - Random seed used for reproducibility (user-specified or generated).
    %   bestTrainedNet - The best trained neural network found during optimization.

    if nargin < 3
        randomSeed = randi(10000);
    else
        randomSeed = userSeed;
    end

    tic;
    disp("Optimization exploration_02 in progress. This process may take up to 30 seconds...");

    trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', 'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
    activationFunctionOptions = {'tansig', 'logsig'};

    gaOptions = optimoptions('ga', ...
        'PopulationSize', 17, ...
        'MaxGenerations', 13, ...
        'CrossoverFraction', 0.8, ...
        'ConstraintTolerance', 0.000991, ...
        'FitnessLimit', 0.000991, ...
        'EliteCount', 2, ...
        'Display', 'iter', ...
        'UseParallel', true);

    global bestTrainedNet;
    bestTrainedNet = [];
    global nnPerfVect;
    nnPerfVect = [];

    function [avgMSEs] = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed); 

        hiddenLayer1Size = round(hyperParams(1)); 
        hiddenLayer2Size = round(hyperParams(2)); 
        maxEpochs = round(hyperParams(3)); 
        trainingFunctionIdx = round(hyperParams(4));
        activationFunctionIdx = round(hyperParams(5));

        net = feedforwardnet([hiddenLayer1Size, hiddenLayer2Size], trainingFunctionOptions{trainingFunctionIdx});
        net.trainParam.showWindow = false;
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};
        net.layers{2}.transferFcn = activationFunctionOptions{activationFunctionIdx};

        perfMetrics = zeros(1, 3);
        mseValues = zeros(1, 1);
        numTrials = 6;

        for trial = 1:numTrials
            net = init(net);
            [net, tr] = train(net, x, t);
            predictedValues = net(x);
            mseValues(trial) = mean((t - predictedValues).^2);
            perfMetrics(trial, :) = [tr.perf(end), tr.vperf(end), tr.tperf(end)];
        end

        avgMSEs = mean(mseValues); 
        avgPerfMetrics = mean(perfMetrics, 1);

        if isempty(bestTrainedNet) || avgMSEs < bestTrainedNet.mse
            bestTrainedNet.mse = avgMSEs;
            bestTrainedNet.net = net;
        end

        nnPerfVect = avgPerfMetrics;
    end

    fitnessFunction = @(hyperParams) evaluateHyperparameters(hyperParams, x, t, randomSeed);
    optimalHyperParams = ga(fitnessFunction, 5, [], [], [], [], lowerBounds, upperBounds, [], gaOptions);
    finalMSE = evaluateHyperparameters(optimalHyperParams, x, t, randomSeed);
end

function plotResults(Qa, Ha, removedQH, QH_beps, D_beps, bestTrainedNetH, diameterToRemove, distinctDiameters, iteration, optimalHyperParamsH, mseDiameter, error)
    figure(1);
    hold on;
    plot(Qa, Ha, 'o', 'MarkerSize', 5, 'LineWidth', 2);
    plot(removedQH(1, :), removedQH(2, :), 'xr', 'MarkerSize', 10, 'LineWidth', 2);
    predictedH = bestTrainedNetH([Qa; diameterToRemove])';
    plot(Qa, predictedH, '--g', 'LineWidth', 2);
    title(sprintf('Diameter: %d, Iteration: %d, Params: [%d %d %d %d %d], MSE: %.6f, Errors: [%.4f, %.4f, %.4f]', diameterToRemove, iteration, optimalHyperParamsH, mseDiameter, error));
    xlabel('Q Flowrate');
    ylabel('H Head');
    legend('Original Data', 'Removed Data', 'Predicted Data');
    hold off;

    figure(2);
    hold on;
    plot(QH_beps(1,:), QH_beps(2,:), 'o', 'MarkerSize', 5, 'LineWidth', 2);
    predictedH_beps = bestTrainedNetH([QH_beps(1,:); D_beps])';
    plot(QH_beps(1,:), predictedH_beps, '--g', 'LineWidth', 2);
    title(sprintf('Diameter: %d, Iteration: %d, MSE for QH_{beps}: %.6f', diameterToRemove, iteration, mseDiameter));
    xlabel('Q Flowrate');
    ylabel('H Head');
    legend('QH_{beps}', 'Predicted QH_{beps}');
    hold off;
end

function writeResults(result, filename)
    % Write results to a CSV file
    resultTable = array2table(result, 'VariableNames', ...
        {'Iteration', 'HiddenLayer1', 'HiddenLayer2', 'Epochs', 'TrainFcn', 'ActivationFcn', 'FinalMSE', 'RandomSeed', 'TrainMSE', 'ValMSE', 'TestMSE'});
    writetable(resultTable, filename);
end
