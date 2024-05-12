%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author : Mohammed twheed khater
%   email: mohammedtwheed@gmail.com
% Submited to : prof. dr. mohammed farid khalil
% its related to trimming part of our bachelor project with dr. farid
% our supervisor.
% 30/4/2023
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% This code defines a program for training two separate neural networks to
% predict the diameter and power consumption of a centrifugal pump impeller
% after trimming based on the flow rate and head. Here's a breakdown of the
% code:
%
% Functions:
%
% 1. `loadData`: This function loads data from four separate `.mat` files
% containing flow rate (Q), head (H), diameter (D), and power (P) values.
% It performs error handling to ensure the data path exists and loads the
% data into MATLAB variables.
%
% 2. `optimizeNNForTrimmingPumpImpeller`: This function performs the core
% task of optimizing a neural network for a given input-output pair (either
% Q,H predicting D or Q,D predicting P). It uses a genetic algorithm to
% search for the best hyperparameters (like hidden layer size, training
% epochs, etc.) that minimize the mean squared error (MSE) between the
% network's predictions and the actual values. It also logs the
% optimization results to a file called
% 'optimizeNNForTrimmingPumpImpeller_log.txt'.
%
% 3. `processDataAndVisualize`: This function processes the data by
% interpolating it using `griddata` to create a smooth surface. It then
% visualizes the results using `mesh` plots for both predicted diameters
% and power consumption from the trained neural networks. Additionally, it
% overlays the original data points on the plots for comparison. Finally,
% it saves the visualizations as `.fig` and `.png` files (if the optional
% argument `saveFigures` is set to `true`).
%
% Overall Process:
%
% 1. The program starts by loading the data from the `.mat` files using
% `loadData`.
% 2. It then optimizes two separate neural networks:
%     - One network for predicting the diameter (D) based on the flow rate
%     (Q) and head (H).
%     - Another network for predicting the power
%     consumption (P) based on the flow rate (Q) and diameter (D).
% 3. Finally, it processes the data and visualizes the results, including
% the neural network predictions and the original data points.
%
% Key Points:
%
% - The code utilizes genetic algorithms for hyperparameter optimization,
% making it an automated approach to finding the best network
% configuration.
% - Separate networks are trained for diameter and power
% prediction due to the difference in the number of data points available
% for each task.
% - Data interpolation is performed to create a smoother
% surface for visualization purposes.
% - The code includes functionalities
% for error handling, logging optimization results, and saving
% visualizations.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NOTES
% special attention to fact that :
%
% - load('QH.mat')
%     - loads the variable QH 331x2 double in the code
% - load('D.mat'):
%     - loads the variable D 331x1 double in the code
% - load('QD.mat'):
%     - loads the variable QD 656x2 double in the code
% - load('Pow.mat'):
%     - loads the variable P 656x1 double in the code
%
% where :
%
%     QH contains the Q flowrate and Head data corresponding to D the
%     diameters corresponding to the (Q ,H).
%
%     QD contains Q flowrate and D diameter corresponding to power in P.
%
% - out main purpose is to build a neural network that takes in (Q,H) and
% gives (D,P) but due the inconsistency in data points number we split it
% to two networks.
%
% - for each neural network it needs to take `input training data` as
% (number of inputs x number of examples or points) this is the reason
% why you will find some wired transposes we use in our code
% since for example the ploting functions will need to take the data
% as column vectors while the neural network traning function takes the
% transpose of this.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN
clear; clc; clf;
% Set data path (replace with your actual data directory if you use our
% trimming.zip folder leave it as it is)
dataPath = '../training-data';

% Load data with error handling
try
    [QH, D, QD, P] = loadData(dataPath);
catch ME
    disp(ME.message);
    return;
end

% % % User-specified random seed (optional)
% % % Replace with your desired seed (or leave empty)
userSeed = 12345;
% %
% % % Optimize neural network for diameter trimming
% [optimalHyperParamsD,finalMSED, randomSeedD, bestTrainedNetD] = ...
%     optimizeNNForTrimmingPumpImpeller(QH, D, userSeed);
% %
% % % Optimize neural network for power consumption
% [optimalHyperParamsP, finalMSEP, randomSeedP, bestTrainedNetP] = ...
%     optimizeNNForTrimmingPumpImpeller(QD, P, userSeed);








Qa = QH(1,:)
Ha = QH(2,:)

% Find indices of value 240 in D
indicesToRemove = find(D == 240);

% Remove rows from QH based on the indices
QH(:,indicesToRemove) = [];
D(:,indicesToRemove)=[];


Q = QH(1,:)
H = QH(2,:)

for i = 1:50

    [optimalHyperParamsH, finalMSEH, randomSeedH, bestTrainedNetH] = ...
        optimizeNNForTrimmingPumpImpeller([QH(1,:);D], QH(2,:), userSeed+i);

  % % Store resultfor this iteration
  result(i,:) = [i, optimalHyperParamsH, finalMSEH, randomSeedH];

% Define desired diameter values 
desiredDiameters = [220,240,250,260];  


% Create a single figure for all plots
figure;
hold on;  % Keep plots on the same figure


Qt=linspace(0,400, 200);

for diameterIndex = 1:length(desiredDiameters)

  
    desiredDiameter = desiredDiameters(diameterIndex);
    Dt =  repmat(desiredDiameter,length(Qt), 1);

    filteredQH = bestTrainedNetH([Qt;Dt'])';

    % Define legend entry for each diameter
     legendLabel = strcat('Diameter: ', num2str(desiredDiameter), 'mm');
  
    % Plot Q vs H with appropriate label and legend entry
    scatter(Qt, filteredQH, 'filled', 'DisplayName', legendLabel);



    scatter(Qa', Ha');

  

end


xlabel('Q (m^3/h)');
ylabel('H (m)');
title('(Q,H) slices with Diameters');
legend;  
hold off;

  % Save the plot with a descriptive filename, as seif noted its done.

  filename = sprintf('./loop_01/dr-farid-edit_loop_%d_%d-%d-%d-%d-%d.png',i, optimalHyperParamsH(1),...
    optimalHyperParamsH(2), optimalHyperParamsH(3), optimalHyperParamsH(4),optimalHyperParamsH(5));
  saveas(gcf, filename);

  % Close the figure to avoid memory issues, as seif noted its done.
  close(gcf);

end



% Write the resultto a CSV file
writematrix([["Iteration", "Hidden Layer 1 Size","Hidden Layer 2 Size", "Max Epochs",...
    "Training Function", "Activation Function", "Final MSE", ...
    "Random Seed"]; result], 'results_loop.csv');

disp('resultsaved to results_loop.csv');



% % Process data and visualize results
% %PLEASE note the transpose here
% processDataAndVisualize(QH', D', QD',P', bestTrainedNetD, bestTrainedNetP);

% % it will save the figure as matlab figure and as a png to include
% % in publications directly




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





function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed)
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
lowerBounds = [1,  1,    50, 1, 1];
upperBounds = [23,29,    200, 3, 2];

% Define options for the genetic algorithm. ConstraintTolerance, is the
% convergence limit its value determines the stop criteria for the ga.
% FitnessLimit, is the min mse value to stop search if mse get to it.
% gaOptions = optimoptions('ga', 'MaxTime',
% 20,'ConstraintTolerance',0.0003,'FitnessLimit',0.0009);
gaOptions = optimoptions('ga', 'MaxTime', 2);

% Global variable to store the best trained neural network found during
% optimization.
global bestTrainedNet;
bestTrainedNet = [];
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
    function avgMSEs = evaluateHyperparameters(hyperParams, x, t, randomSeed)
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
        end
    end

% Set a random seed for reproducibility.
randomSeed = randi(10000);
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



function processDataAndVisualize(QH, D, QD, P, bestTrainedNetD, bestTrainedNetP,saveFigures)
% processDataAndVisualize - Process data and visualize results.
%   Inputs: - QH: Q flowrate and Head data. - D: Diameters. - QD: Q
%   flowrate and diameters. - P: Power values. - bestTrainedNetD: Best
%   trained neural network for diameters. - bestTrainedNetP: Best trained
%   neural network for power.

% Data processing involves interpolating data points and preparing for
% visualization. Visualization includes plotting data points and neural
% network predictions. Separate visualizations are created for diameters
% and power.
% saveFigures (optional): Boolean flag indicating whether to save
% figures (default: true).

if ~exist('saveFigures','var')
    % third parameter does not exist, so default it to something
    saveFigures = true;
end



% Data interpolation for diameters

% PLEASE note the transpose in
% sim(bestTrainedNetD, QH') and sim(bestTrainedNetP, QD')
% and for griddata it needs column vectors
% in general processDataAndVisualize is assumed to take in
% column vectors but for neural networks it need to be transposed

[Qq, Hq] = meshgrid(0:2:440, 0:.5:90);
Dq = griddata(QH(:,1), QH(:,2), sim(bestTrainedNetD, QH'), Qq, Hq);

% Data interpolation for power
[QDq, Pq] = meshgrid(0:2:440, 220:.5:270);
DqP = griddata(QD(:,1), QD(:,2), sim(bestTrainedNetP, QD'), QDq, Pq);

% Visualization of diameters
figure;
subplot(2, 1, 1);
mesh(Qq, Hq, Dq);
xlabel('Flow Rate (m^3/h)');
ylabel('Head (m)');
zlabel('Diameter (mm)');
title('Neural Network vs Data Points (Diameters)');

% Scatter plot of data points
hold on;
scatter3(QH(:,1), QH(:,2), D, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
legend('Neural Network', 'Data Points');
hold off;

% Visualization of power
subplot(2, 1, 2);
mesh(QDq, Pq, DqP);
xlabel('Flow Rate (m^3/h)');
ylabel('Diameter (mm)');
zlabel('Power in (kW)');
title('Neural Network vs Data Points (Power)');

% Scatter plot of data points
hold on;
scatter3(QD(:,1), QD(:,2), P, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
legend('Neural Network', 'Data Points');
hold off;


% Save figures based on optional argument
if saveFigures
    currentTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');


    saveas(gcf, ['diameter_power_visualization_' char(currentTime) '.fig']);
    saveas(gcf, ['diameter_power_visualization_' char(currentTime) '.png']);
end
end


