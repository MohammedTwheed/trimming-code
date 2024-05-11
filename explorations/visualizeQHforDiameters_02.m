clear; clc; clf;

load('..\training-data\QH.mat');
load('..\training-data\D.mat');




% % Find indices of value 240 in D
% indicesToRemove = find(D == 240);

% % Remove rows from QH based on the indices
% QH(indicesToRemove, :) = [];
% D(indicesToRemove, :)=[];


Q = QH(:,1);
H = QH(:,2);

[finalMSE_H,  bestTrainedNet_H] ...
    = optimizeNNForTrimmingPumpImpeller([Q';D' ],H');




% Define desired diameter values 
desiredDiameters = [220,240,245,250,255,260];  


% Create a single figure for all plots
figure;
hold on;  % Keep plots on the same figure


Qt=linspace(0,400, 200);

for diameterIndex = 1:length(desiredDiameters)

  
    desiredDiameter = desiredDiameters(diameterIndex);
    Dt =  repmat(desiredDiameter,length(Qt), 1);

    filteredQH = bestTrainedNet_H([Qt;Dt'])';

  % Plot Q vs H with appropriate label
  scatter(Qt, filteredQH, 'filled');
  legendStr = sprintf('D = %.2f', desiredDiameter);
  legend_handle(diameterIndex) = plot(NaN, NaN, 'DisplayName', legendStr);  % Placeholder for legend



    scatter(QH(:,1), QH(:,2));

  

end



  
  

  

% Customize plot appearance
xlabel('Q (m^3/h)');
ylabel('H (m)');
title('(Q,H) slices with Diameters');  % Adjust title as needed
legend(legend_handle);  % Add legend with diameter labels
hold off;

  % Save the plot with a descriptive filename, as seif noted its done.
  optimalHyperParams= [9,1, 200, 1, 1];
  filename = sprintf('dr-farid-edit_%d-%d-%d-%d.png', optimalHyperParams(1),...
      optimalHyperParams(2), optimalHyperParams(3), optimalHyperParams(4));
  saveas(gcf, filename);

  % Close the figure to avoid memory issues, as seif noted its done.
  close(gcf);



% function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed)
%     % optimizeNNForTrimmingPumpImpeller: Optimizes neural network
%     % hyperparameters for pump impeller trimming. Inputs:
%     %   x - Input data (Q flowrate, H head) for the neural network. t -
%     %   Target data (D diameter or power but not both at same time) for the
%     %   neural network. userSeed (optional) - User-specified random seed for
%     %   reproducibility.
%     % Outputs:
%     %   optimalHyperParams - Optimized hyperparameters found by the genetic
%     %   algorithm. finalMSE - Mean squared error (MSE) of the best model.
%     %   randomSeed - Random seed used for reproducibility (user-specified or
%     %   generated). bestTrainedNet - The best trained neural network found
%     %   during optimization.
    
%     % % TODO: Validate input data dimensions
%     % if ~isequal(size(x, 1), size(t, 1))
%     %   error('Input and target data must have the same number of samples.');
%     % end
    
%     % % TODO: Validate hyperparameter bounds if any(lowerBounds >=
%     % upperBounds)
%     %   error('Lower bounds must be less than upper bounds for
%     %   hyperparameters.');
%     % end
    
%     % Set random seed (user-specified or generated)
%     if nargin < 3
%         randomSeed = randi(10000);
%     else
%         randomSeed = userSeed;
%     end
    
%     % Start timer to measure the duration of the optimization process.
%     tic;
%     disp("Optimization exploration_02 in progress. This process may take up to 30 seconds...");
    
%     % Define the available options for training functions and activation
%     % functions.
%     trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', ...
%         'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
%     activationFunctionOptions = {'tansig', 'logsig'};
    
%     % Define bounds for the genetic algorithm optimization. the positional
%     % meaning [<hidden layer neurons number> ,< epochs>... ,<index of
%     % trainingFunctionOptions>,<index of activationFunctionOptions>]
%     lowerBounds = [90,  74,    50,1, 1];
%     upperBounds = [91,75, 51, 2, 2];
    
%     % Define options for the genetic algorithm. ConstraintTolerance, is the
%     % convergence limit its value determines the stop criteria for the ga.
%     % FitnessLimit, is the min mse value to stop search if mse get to it.
%     % gaOptions = optimoptions('ga', 'FitnessLimit',0.0008,'ConstraintTolerance',0.0002,'MaxTime', 20);
%     gaOptions = optimoptions('ga', 'MaxTime', 20);
    
%     % Global variable to store the best trained neural network found during
%     % optimization.
%     global bestTrainedNet;
%     bestTrainedNet = [];
%     % TODO: MTK to SEI you might consider making a helper function just to
%     % resolve this issue with the ga for example function
%     % [mse,bestTrainedNet] = evaluateHyperparameters(params...)
%     %  end
%     % function  mse = f_ga(params...) [mse,bestTrainedNet] =
%     % evaluateHyperparameters(params...)
%     %  end
%     % but would this leak the bestTrainedNet #NeedResearch ??!! monadic
%     % approach to side effects as in haskell SEI: just re train it MTK:!!
    
    
%     % evaluateHyperparameters is a local function to evaluate hyperparameters
%     % using the neural network. it builds the neural net and train it then
%     % get the average mean square of training, validation and testing
%     % datasets errors As we divide our data by ratio 70% training , 15%
%     % validation , 15% testing. all ga will optimize based on that avgMSEs
%     % returned by evaluateHyperparameters.
%         function avgMSEs = evaluateHyperparameters(hyperParams, x, t, randomSeed)
%             rng(randomSeed); % Set random seed for reproducibility.
    
%             % Extract hyperparameters.
%             hiddenLayer1Size = round(hyperParams(1)); %Hidden Layer Size
%             hiddenLayer2Size = round(hyperParams(2)); %Hidden Layer Size
%             maxEpochs = round(hyperParams(3));       %Max Epochs
%             %Training Function
%             trainingFunctionIdx = round(hyperParams(4));
%             %Activation Function or transfere function
%             activationFunctionIdx = round(hyperParams(5));
%             % Define the neural network.
%             net = feedforwardnet([hiddenLayer1Size,hiddenLayer2Size],...
%                 trainingFunctionOptions{trainingFunctionIdx});
%             % Suppress training GUI for efficiency.
%             net.trainParam.showWindow = false;
%             net.trainParam.epochs = maxEpochs;
%             net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};
%             net.layers{2}.transferFcn = activationFunctionOptions{activationFunctionIdx};
    
%             % Choose a Performance Function
%             net.performFcn = 'mse';
    
    
%             % Choose Input and Output Pre/Post-Processing Functions
%             net.input.processFcns = {'removeconstantrows', 'mapminmax'};
%             net.output.processFcns = {'removeconstantrows', 'mapminmax'};
    
%             % Define data split for training, validation, and testing.
    
%             % For a list of all data division functions type: help nndivide
%             net.divideFcn = 'dividerand';  % Divide data randomly
%             net.divideMode = 'sample';  % Divide up every sample
%             net.divideParam.trainRatio = 0.7;
%             net.divideParam.valRatio = 0.15;
%             net.divideParam.testRatio = 0.15;
    
    
%             % Train the neural network.
%             [trainedNet, tr] = train(net, x, t);
    
%             % Evaluate the model performance using mean squared error (MSE).
    
%             % predictions = trainedNet(normalized_input);
%             predictions = trainedNet(x);
%             mse = perform(trainedNet, t, predictions);
    
%             % Recalculate Training, Validation and Test Performance
%             trainTargets        = t .* tr.trainMask{1};
%             valTargets          = t .* tr.valMask{1};
%             testTargets         = t .* tr.testMask{1};
%             trainPerformance    = perform(net,trainTargets,predictions);
%             valPerformance      = perform(net,valTargets,predictions);
%             testPerformance     = perform(net,testTargets,predictions);
    
%             % for better performance we came up with this convention rather than
%             % using the mse based on perform function only
%             avgMSEs = (mse +  ...
%                 trainPerformance +...
%                 valPerformance+....
%                 testPerformance) / 4;
    
    
%             % Check if the current MSE is the best MSE so far and update the
%             % global variable if necessary.
%             if isempty(bestTrainedNet) || avgMSEs < perform(bestTrainedNet, ...
%                     t, bestTrainedNet(x))
%                 bestTrainedNet = trainedNet;
%             end
%         end
    
%     % Set a random seed for reproducibility.
%     randomSeed = randi(10000);
%     rng(randomSeed);
    
%     % Perform optimization using genetic algorithm.
%     [optimalHyperParams, finalMSE] = ga(@(hyperParams)evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
%         length(lowerBounds), [], [], [], [], lowerBounds, upperBounds, [], gaOptions);
    
%     % Round the optimized hyperparameters to integers.
%     optimalHyperParams = round(optimalHyperParams);
    
%     % Measure elapsed time.
%     elapsedTime = toc;
    
%     % Extract the chosen training and activation functions.
%     trainingFunction = trainingFunctionOptions{optimalHyperParams(4)};
%     activationFunction = activationFunctionOptions{optimalHyperParams(5)};
    
%     % Logging functionality SIH
%     logFile = 'optimizeNNForTrimmingPumpImpeller_log.txt';
    
%     % Open log file for appending (create if it doesn't exist)
%     fid = fopen(logFile, 'a');
%     if fid == -1
%         error('Error opening log file for appending.');
%     end
    
%     % Write current timestamp to log file
%     currentTime = datetime('now', 'Format', 'yyyy-MM-dd HH:MM:SS');
%     fprintf(fid, '%s\n', char(currentTime));
    
%     % Write optimization results to log file
%     fprintf(fid, 'Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
%         optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3),...
%         trainingFunction, activationFunction);
%     fprintf(fid, 'Final Mean Squared Error (MSE): %.4f\n', finalMSE);
%     fprintf(fid, 'Random Seed Used: %d\n', randomSeed);
%     fprintf(fid, 'Optimization Duration: %.4f seconds\n\n', elapsedTime);
    
%     % Close the log file
%     fclose(fid);
    
    
%     % % Logging functionality with enhancements MTK short formate
%     % logFilename = 'optimizeNNForTrimmingPumpImpeller_log.txt';
%     %
%     % % Open log file for appending (create if it doesn't exist)
%     % fid = fopen(logFilename, 'a');
%     % if fid == -1
%     %   error('Error opening log file for appending.');
%     % end
%     %
%     % % Write header line to log file if it doesn't exist
%     % if ftell(fid) == 0
%     %   fprintf(fid, 'Date & Time\tHidden Layer 1 Size\tHidden Layer 2 Size\tMax Epochs\tTraining Function\tActivation Function\tFinal MSE\tRandom Seed\tOptimization Time (seconds)\n');
%     % end
%     %
%     % % Create cell array for logging data
%     % logData = {datetime('now', 'Format', 'yyyy-MM-dd HH:mm:SS'), ...
%     %            optimalHyperParams(1), optimalHyperParams(2), ...
%     %            optimalHyperParams(3), trainingFunction, activationFunction, ...
%     %            finalMSE, randomSeed, elapsedTime};
%     %
%     % % Write data to log file in a single fprintf call
%     % fprintf(fid, '%s\t%d\t%d\t%d\t%s\t%s\t%.4f\t%d\t%.4f\n', logData{:});
%     %
%     % % Close the log file
%     % fclose(fid);
    
    
%     % results.
%     fprintf('Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d,Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
%         optimalHyperParams(1), optimalHyperParams(2),optimalHyperParams(3),...
%         trainingFunction, activationFunction);
%     fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
%     fprintf('Random Seed Used: %d\n', randomSeed);
%     fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);
    
%     end



function [ finalMSE, bestTrainedNet] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed)
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
    lowerBounds = [90,  74,    50,1, 1];
    upperBounds = [91,75, 51, 2, 2];
    
    % Define options for the genetic algorithm. ConstraintTolerance, is the
    % convergence limit its value determines the stop criteria for the ga.
    % FitnessLimit, is the min mse value to stop search if mse get to it.
    % gaOptions = optimoptions('ga', 'FitnessLimit',0.0008,'ConstraintTolerance',0.0002,'MaxTime', 20);
    gaOptions = optimoptions('ga', 'MaxTime', 20);
    
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
            net = feedforwardnet([hiddenLayer1Size,hiddenLayer2Size],trainingFunctionOptions{trainingFunctionIdx});
            % Suppress training GUI for efficiency.
            % net.trainParam.showWindow = false;
            % net.trainParam.epochs = maxEpochs;
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
    
    

                bestTrainedNet = trainedNet;
          
        end
    
    % Set a random seed for reproducibility.
    randomSeed = randi(10000);
    rng(randomSeed);
    
    % Perform optimization using genetic algorithm.
    finalMSE= evaluateHyperparameters([9,1,200, 1, 1], x, t, randomSeed);

    
    end
    
    
    
