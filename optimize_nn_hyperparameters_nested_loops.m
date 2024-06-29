%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN
clear; clc; clf; close all;
load('training-data/filtered_QHD_table.mat')
load('training-data/filtered_QDP_table.mat')
load('training-data/deleted_QHD_table.mat')
load('training-data/deleted_QDP_table.mat')

QH = [filtered_QHD_table.FlowRate_m3h,filtered_QHD_table.Head_m]';
D  = [filtered_QHD_table.Diameter_mm]';

QH_beps=[deleted_QHD_table.FlowRate_m3h,deleted_QHD_table.Head_m]';
D_beps=[deleted_QHD_table.Diameter_mm]';

QD = [filtered_QDP_table.FlowRate_m3h,filtered_QDP_table.Diameter_mm]';
P = [filtered_QDP_table.Power_kW]';

QD_beps=[deleted_QDP_table.FlowRate_m3h,deleted_QDP_table.Diameter_mm]';
P_beps=[deleted_QDP_table.Power_kW]';

% User-specified random seed (optional)
% Replace with your desired seed (or leave empty)
userSeed = 4826;

% Define a threshold for MSE to exit the loop early
mseThreshold = 0.00199;

% Find all distinct diameters in D
distinctDiameters = unique(D);

% Weights for combining MSEs
weightDiameter = 0.5;
weightBeps = 0.5;

% Define the filename for results
resultsFilename = './optimize_nn_hyperparameters_nested_loops/results_loop.csv';
structFilename = './optimize_nn_hyperparameters_nested_loops/results_struct.mat';

% Check if the file exists to determine if headers need to be written
if ~isfile(resultsFilename)
    % Write the header to the CSV file
    writematrix([ "Diameter Removed", "Iteration", "Hidden Layer 1 Size", "Hidden Layer 2 Size", "Max Epochs", ...
        "Training Function", "Activation Function", "Final MSE", ...
        "Random Seed", "Training Error", "Validation Error", "Test Error",'mseQH_beps'], resultsFilename);
end

% Initialize an empty struct array
resultsStruct = struct('Timestamp', {}, 'DiameterRemoved', {}, 'Iteration', {}, 'HiddenLayer1Size', {}, 'HiddenLayer2Size', {}, ...
    'MaxEpochs', {}, 'TrainingFunction', {}, 'ActivationFunction', {}, 'FinalMSE', {}, 'RandomSeed', {}, 'TrainingError', {}, ...
    'ValidationError', {}, 'TestError',{},'mseQH_beps', {}, 'TrainedNet', {}, 'FigureFileName', {});


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
    lower_bounds=[2,    9,    100, 1, 1];
    upper_bounds=[17,  95,    300, 1, 1];
    
    % Track the previous combined MSE to determine the improvement
    prevCombinedMSE = inf;

    for i = 1:30
        [optimalHyperParamsH, finalMSEH, randomSeedH, bestTrainedNetH, error] = ...
            optimizeNNForTrimmingPumpImpeller([QH_temp(1,:); D_temp], QH_temp(2,:), userSeed+i, lower_bounds, upper_bounds);

        % Calculate MSE for the removed diameter
        predictedH = bestTrainedNetH([removedQH(1, :); removedD])';
        mseDiameter = mean((removedQH(2, :)' - predictedH).^2 / sum(removedQH(2, :)));

        predictedH_beps = bestTrainedNetH([QH_beps(1,:); D_beps])';
        mseQH_beps = mean((QH_beps(2,:)' - predictedH_beps).^2 / sum(QH_beps(2,:)));

        fprintf('Diameter %d, Iteration %d, MSE_Dia: %.6f,  MSE_beps: %.6f \n', diameterToRemove, i, mseDiameter, mseQH_beps);

        % % Combine the two MSEs into a single metric
        % combinedMSE = weightDiameter * mseDiameter + weightBeps * mseQH_beps;
        % 
        % % Determine the change in combined MSE
        % deltaMSE = prevCombinedMSE - combinedMSE;
        % 
        % % Adjust the bounds based on the improvement in combined MSE
        % if deltaMSE > 0.01  % Significant improvement
        %     adjustment = [3, 3, 13, 0, 0];
        % elseif deltaMSE > 0.001  % Moderate improvement
        %     adjustment = [2, 2, 17, 0, 0];
        % else  % Minor improvement
        %     adjustment = [1, 1, 7, 0, 0];
        % end
        % 
        % lower_bounds = max(lower_bounds, [2,optimalHyperParamsH(2),optimalHyperParamsH(3),1,1] - adjustment);
        % upper_bounds = min(upper_bounds, [2,optimalHyperParamsH(2),optimalHyperParamsH(3),2,1] + adjustment);
        % 
        % % Update the previous combined MSE for the next iteration
        % prevCombinedMSE = combinedMSE;

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
        filename = sprintf('./optimize_nn_hyperparameters_nested_loops/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%.6f_test-%.6f_mseQH_beps-%6f.png', diameterToRemove, i, ...
            optimalHyperParamsH(1), optimalHyperParamsH(2), optimalHyperParamsH(3), ...
            optimalHyperParamsH(4), optimalHyperParamsH(5), mseDiameter, error(3),mseQH_beps);
        saveas(gcf, filename);

        % Specify the filename for saving the network
        filename = sprintf('./optimize_nn_hyperparameters_nested_loops/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%.6f_test-%.6f.mat', diameterToRemove, i, ...
            optimalHyperParamsH(1), optimalHyperParamsH(2), optimalHyperParamsH(3), ...
            optimalHyperParamsH(4), optimalHyperParamsH(5), mseDiameter, error(3));
            
        % Create a new entry in the results struct
        newEntry = struct('Timestamp', datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'), 'DiameterRemoved', diameterToRemove, 'Iteration', i, ...
            'HiddenLayer1Size', optimalHyperParamsH(1), 'HiddenLayer2Size', optimalHyperParamsH(2), 'MaxEpochs', optimalHyperParamsH(3), ...
            'TrainingFunction', optimalHyperParamsH(4), 'ActivationFunction', optimalHyperParamsH(5), 'FinalMSE', finalMSEH, ...
            'RandomSeed', randomSeedH, 'TrainingError', error(1), 'ValidationError', error(2), 'TestError', error(3),'mseQH_beps',mseQH_beps, ...
            'TrainedNet', bestTrainedNetH, 'FigureFileName', filename);

        % Append the new entry to the results struct array
        resultsStruct(end+1) = newEntry;

        % Save the struct array to a .mat file
        save(structFilename, 'resultsStruct');
        % Close the figure to avoid memory issues
        close(gcf);

        % Append the result to the CSV file

        newResult = [diameterToRemove, i, optimalHyperParamsH, finalMSEH, randomSeedH, error(1), error(2), error(3),mseQH_beps];
        writematrix(newResult, resultsFilename, 'WriteMode', 'append');

        % Exit loop if MSE is below the threshold
        if (mseDiameter < mseThreshold) && (error(3) < 0.0199) && (mseQH_beps < mseThreshold)
            fprintf('MSE for diameter %d is below the threshold. Exiting loop.\n', diameterToRemove);
            break;
        end
    end
end

disp('Results saved to optimize_nn_hyperparameters_nested_loops/results_loop.csv');

% END MAIN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








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





function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet,nnPerfVect] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed,lowerBounds,upperBounds)
% optimizeNNForTrimmingPumpImpeller: Optimizes neural network
% hyperparameters for pump impeller trimming. Inputs:
%   x - Input data (Q flowrate, H head) for the neural network. t -
%   Target data (D diameter or power but not both at same time) for the
%   neural network. userSeed (optional) - User-specified random seed for
%   reproducibility.
% Outputs:
%   optimalHyperParams - Optimized hyperparameters found by the genetic
%   algorithm. finalMSE - Mean squared error (MSE) of the best model.
%   randomSeed - Random seed used for reproducibility (user-specified or
%   generated). bestTrainedNet - The best trained neural network found
%   during optimization.

% % TODO: Validate input data dimensions
% if ~isequal(size(x, 1), size(t, 1))
%   error('Input and target data must have the same number of samples.');
% end

% % TODO: Validate hyperparameter bounds if any(lowerBounds >=
% upperBounds)
%   error('Lower bounds must be less than upper bounds for
%   hyperparameters.');
% end

% Set random seed (user-specified or generated)
if nargin < 3
    randomSeed = randi(10000);
else
    randomSeed = userSeed;
end

% Start timer to measure the duration of the optimization process.
tic;
disp("Optimization of nn in progress. This process may take up to 1 or 2 hours using my default settings...");

% Define the available options for training functions and activation
% functions.
trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', ...
    'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
activationFunctionOptions = {'tansig', 'logsig'};

% Define bounds for the genetic algorithm optimization. the positional
% meaning [<hidden layer neurons number> ,< epochs>... ,<index of
% trainingFunctionOptions>,<index of activationFunctionOptions>]


% lowerBounds = [2,  60,    117, 1, 1];
% upperBounds = [2,  117,    1000, 1, 1];

% Define options for the genetic algorithm. ConstraintTolerance, is the
% convergence limit its value determines the stop criteria for the ga.
% FitnessLimit, is the min mse value to stop search if mse get to it.


% gaOptions = optimoptions('ga','ConstraintTolerance',0.0009,'FitnessLimit',0.0009,'MaxTime',...
% 20);

gaOptions=optimoptions('ga', ...
    'PopulationSize', 17, ...
    'MaxGenerations', 13, ...
    'CrossoverFraction', 0.8, ...
    'ConstraintTolerance',0.000991, ...
    'FitnessLimit',0.000991, ...
    'EliteCount', 2, ...
    'Display', 'iter', ...
    'UseParallel', true);

% gaOptions = optimoptions('ga', 'MaxTime', 2);

% Global variable to store the best trained neural network found during
% optimization.
global bestTrainedNet;
bestTrainedNet = [];
global nnPerfVect;
nnPerfVect=[];
% TODO: MTK to SEI you might consider making a helper function just to
% resolve this issue with the ga for example function
% [mse,bestTrainedNet] = evaluateHyperparameters(params...)
%  end
% function  mse = f_ga(params...) [mse,bestTrainedNet] =
% evaluateHyperparameters(params...)
%  end
% but would this leak the bestTrainedNet #NeedResearch ??!! monadic
% approach to side effects as in haskell SEI: just re train it MTK:!!


% evaluateHyperparameters is a local function to evaluate hyperparameters
% using the neural network. it builds the neural net and train it then
% get the average mean square of training, validation and testing
% datasets errors As we divide our data by ratio 70% training , 15%
% validation , 15% testing. all ga will optimize based on that avgMSEs
% returned by evaluateHyperparameters.
    function [avgMSEs] = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed); % Set random seed for reproducibility.

        % Extract hyperparameters.
        hiddenLayer1Size = round(hyperParams(1)); %Hidden Layer Size
        hiddenLayer2Size = round(hyperParams(2)); %Hidden Layer Size
        maxEpochs = round(hyperParams(3));       %Max Epochs
        %Training Function
        trainingFunctionIdx = round(hyperParams(4));
        %Activation Function or transfere function
        activationFunctionIdx = round(hyperParams(5));
        % Define the neural network.
        net = feedforwardnet([hiddenLayer1Size,hiddenLayer2Size],...
            trainingFunctionOptions{trainingFunctionIdx});
        % Suppress training GUI for efficiency.
        net.trainParam.showWindow = false;
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};
        net.layers{2}.transferFcn = activationFunctionOptions{activationFunctionIdx};

        % Choose a Performance Function
        net.performFcn = 'mse';


        % Choose Input and Output Pre/Post-Processing Functions
        net.input.processFcns = {'removeconstantrows', 'mapminmax'};
        net.output.processFcns = {'removeconstantrows', 'mapminmax'};

        % Define data split for training, validation, and testing.

        % For a list of all data division functions type: help nndivide
        net.divideFcn = 'dividerand';  % Divide data randomly
        net.divideMode = 'sample';  % Divide up every sample
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;


        % Train the neural network.
        [trainedNet, tr] = train(net, x, t);

        % Evaluate the model performance using mean squared error (MSE).

        % predictions = trainedNet(normalized_input);
        predictions = trainedNet(x);
        mse = perform(trainedNet, t, predictions);

        % Recalculate Training, Validation and Test Performance
        trainTargets        = t .* tr.trainMask{1};
        valTargets          = t .* tr.valMask{1};
        testTargets         = t .* tr.testMask{1};
        trainPerformance    = perform(net,trainTargets,predictions);
        valPerformance      = perform(net,valTargets,predictions);
        testPerformance     = perform(net,testTargets,predictions);

        % for better performance we came up with this convention rather than
        % using the mse based on perform function only
        avgMSEs = (mse +  ...
            trainPerformance +...
            valPerformance+....
            testPerformance) / 4;
        
        

        % Check if the current MSE is the best MSE so far and update the
        % global variable if necessary.
        if isempty(bestTrainedNet) || avgMSEs < perform(bestTrainedNet, ...
                t, bestTrainedNet(x))
            bestTrainedNet = trainedNet;
            nnPerfVect= [trainPerformance,valPerformance,testPerformance];
        end
    end

% Set a random seed for reproducibility.
% randomSeed = randi(userSeed);
rng(randomSeed);

% Perform optimization using genetic algorithm.
[optimalHyperParams, finalMSE] = ga(@(hyperParams)evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
    length(lowerBounds), [], [], [], [], lowerBounds, upperBounds, [], gaOptions);

% Round the optimized hyperparameters to integers.
optimalHyperParams = round(optimalHyperParams);

% Measure elapsed time.
elapsedTime = toc;

% Extract the chosen training and activation functions.
trainingFunction = trainingFunctionOptions{optimalHyperParams(4)};
activationFunction = activationFunctionOptions{optimalHyperParams(5)};

% Logging functionality SIH
logFile = 'optimizeNNForTrimmingPumpImpeller_log.txt';

% Open log file for appending (create if it doesn't exist)
fid = fopen(logFile, 'a');
if fid == -1
    error('Error opening log file for appending.');
end

% Write current timestamp to log file
currentTime = datetime('now', 'Format', 'yyyy-MM-dd HH:MM:SS');
fprintf(fid, '%s\n', char(currentTime));

% Write optimization results to log file
fprintf(fid, 'Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
    optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3),...
    trainingFunction, activationFunction);
fprintf(fid, 'Final Mean Squared Error (MSE): %.4f\n', finalMSE);
fprintf(fid, 'Random Seed Used: %d\n', randomSeed);
fprintf(fid, 'Optimization Duration: %.4f seconds\n\n', elapsedTime);

% Close the log file
fclose(fid);




% results.
fprintf('Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
    optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3),...
    trainingFunction, activationFunction);
fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
fprintf('Random Seed Used: %d\n', randomSeed);
fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

end



