%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN

% Start the timer
tic;
clear; clc; clf;
load('filtered_QHD_table.mat')
load('filtered_QDP_table.mat')
load('deleted_QHD_table.mat')
load('deleted_QDP_table.mat')

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
mseThreshold = 0.000199;
    % Initialize bounds
    lower_bounds=[ 2,     9,       150, 1, 1];
    upper_bounds=[20,      50,       400, 1, 1];
        [optimalHyperParamsH, finalMSEH, randomSeedH, bestTrainedNetH, error] = ...
            optimizeNNForTrimmingPumpImpeller([QD(1,:); QD(2,:)],P,userSeed,lower_bounds,upper_bounds);
        % Store result for this iteration
        result = [optimalHyperParamsH, finalMSEH, randomSeedH, error(1), error(2), error(3)];

elapsedTime = toc;

% Save the elapsed time to a text file
fileID = fopen(      './02/execution_time.txt',         'w');
fprintf(fileID, 'Elapsed Time: %.6f seconds\n', elapsedTime);
fclose(fileID);

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
disp("Optimization exploration_02 in progress. This process may take up to 30 seconds...");

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

gaOptions=optimoptions('ga','ConstraintTolerance',0.000991,'FitnessLimit',0.000991);

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
        avgMSEs = (valPerformance+testPerformance) / 2;
        
        

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


% % Logging functionality with enhancements MTK short formate
% logFilename = 'optimizeNNForTrimmingPumpImpeller_log.txt';
%
% % Open log file for appending (create if it doesn't exist)
% fid = fopen(logFilename, 'a');
% if fid == -1
%   error('Error opening log file for appending.');
% end
%
% % Write header line to log file if it doesn't exist
% if ftell(fid) == 0
%   fprintf(fid, 'Date & Time\tHidden Layer 1 Size\tHidden Layer 2 Size\tMax Epochs\tTraining Function\tActivation Function\tFinal MSE\tRandom Seed\tOptimization Time (seconds)\n');
% end
%
% % Create cell array for logging data
% logData = {datetime('now', 'Format', 'yyyy-MM-dd HH:mm:SS'), ...
%            optimalHyperParams(1), optimalHyperParams(2), ...
%            optimalHyperParams(3), trainingFunction, activationFunction, ...
%            finalMSE, randomSeed, elapsedTime};
%
% % Write data to log file in a single fprintf call
% fprintf(fid, '%s\t%d\t%d\t%d\t%s\t%s\t%.4f\t%d\t%.4f\n', logData{:});
%
% % Close the log file
% fclose(fid);


% results.
fprintf('Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
    optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3),...
    trainingFunction, activationFunction);
fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
fprintf('Random Seed Used: %d\n', randomSeed);
fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

end



