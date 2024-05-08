function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet,probs_out] = optimizeNNForTrimmingPumpImpeller(x, t)
    % Function to optimize neural network hyperparameters for the trimming of a pump impeller.
    % Inputs:
    %   x - Input data (Q flowrate,H head) for the neural network.
    %   t - Target data (D diameter,eta efficiency) for the neural network.
    % Outputs:
    %   optimalHyperParams - Optimized hyperparameters found by the genetic algorithm.
    %   finalMSE - Mean squared error (MSE) of the best model.
    %   randomSeed - Random seed used for reproducibility.
    %   bestTrainedNet - The best trained neural network found during optimization.

    %EXAMPLE USAGE
    % first load the dataset using load('trimming_nn_training_dataset.mat')
    % to find the optimum architecture for example call optimizeNNForTrimmingPumpImpeller
    % [a,b,c,d,probs_out]=optimizeNNForTrimmingPumpImpeller(QH_nn_input',D_eta_nn_output')
    % probs_out is to use with mapminmax('reverse',bestTrainedNet(your
    % input),probs_out) to get scale back since we have normalized it.
    %please make sure you do the transpose if you are using our 'trimming_nn_training_dataset.mat'


    % Start timer to measure the duration of the optimization process.
    tic;
    disp("Optimization in progress. This process may take up to 30 seconds...");

    % Define the available options for training functions and activation functions.
    trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', 'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
    activationFunctionOptions = {'tansig', 'logsig'};

    % Define bounds for the genetic algorithm optimization.
    % the positional meaning [<hidden layer neurons number> ,< epochs>...
    % ,<index of trainingFunctionOptions>,<index of activationFunctionOptions>]
    lowerBounds = [5, 50, 1, 1];
    upperBounds = [200, 200, 8, 2];

    % Define options for the genetic algorithm.
    % ConstraintTolerance, is the convergence limit its value determines
    % the stop criteria for the ga.
    % FitnessLimit, is the min mse value to stop search if mse get to it.
    gaOptions = optimoptions('ga', 'MaxTime', 20,'ConstraintTolerance',0.003,'FitnessLimit',0.009);

    % Global variable to store the best trained neural network found during optimization.
    global bestTrainedNet;
    bestTrainedNet = [];
    % TODO: MTK to SEI you might consider making a helper function 
    % just to resolve this issue with the ga for example 
    % function  [mse,bestTrainedNet] = evaluateHyperparameters(params...)
    %  end
    % function  mse = f_ga(params...) 
    % [mse,bestTrainedNet] = evaluateHyperparameters(params...)
    %  end
    % but would this leak the bestTrainedNet #NeedResearch ??!!
    % monadic approach to side effects as in haskell
    % SEI: just re train it MTK:!!


    % local function to evaluate hyperparameters using the neural network.
    function mse = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed); % Set random seed for reproducibility.

        % Extract hyperparameters.
        hiddenLayerSize = round(hyperParams(1)); %Hidden Layer Size
        maxEpochs = round(hyperParams(2));       %Max Epochs
        trainingFunctionIdx = round(hyperParams(3)); %Training Function
        activationFunctionIdx = round(hyperParams(4));%Activation Function or transfere function

        % Define the neural network.
        net = fitnet(hiddenLayerSize, trainingFunctionOptions{trainingFunctionIdx});
        net.trainParam.showWindow = false; % Suppress training GUI for efficiency.
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};

        % % b.c we don't want to apply mapminmax each time we call 
        % % the function from a script (redundant)
        % net.input.processFcns = {'removeconstantrows', 'mapminmax'};



        % Normalize training data to [-1,1] this the defaul value 
        % for mapminmax
    
        normalized_input = mapminmax(x);
        [normalized_output,probs_out] = mapminmax(t);


        % Define data split for training, validation, and testing.
        net.divideParam.trainRatio = 0.7;
        net.divideParam.valRatio = 0.15;
        net.divideParam.testRatio = 0.15;

        % Train the neural network.
        [trainedNet, ~] = train(net, normalized_input, normalized_output);

        % Evaluate the model performance using mean squared error (MSE).

        predictions = trainedNet(normalized_input);

        % Reverse the normalization process
        predictions = mapminmax('reverse', predictions,probs_out);
        
        % note here the usage of t the un normalized output in the 
        % training dataset
        mse = perform(trainedNet, t, predictions);

        % Check if the current MSE is the best MSE so far and update the global variable if necessary.
        if isempty(bestTrainedNet) || mse < perform(bestTrainedNet, t, bestTrainedNet(x))
            bestTrainedNet = trainedNet;
        end
    end

    % Set a random seed for reproducibility.
    randomSeed = randi(10000);
    rng(randomSeed);

    % Perform optimization using genetic algorithm.
    [optimalHyperParams, finalMSE] = ga(@(hyperParams) evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
        4, [], [], [], [], lowerBounds, upperBounds, [], gaOptions);

    % Round the optimized hyperparameters to integers.
    optimalHyperParams = round(optimalHyperParams);

    % Measure elapsed time.
    elapsedTime = toc;

    % Extract the chosen training and activation functions.
    trainingFunction = trainingFunctionOptions{optimalHyperParams(3)};
    activationFunction = activationFunctionOptions{optimalHyperParams(4)};

    % Display the optimization results.
    fprintf('Optimized Hyperparameters: Hidden Layer Size = %d, Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
        optimalHyperParams(1), optimalHyperParams(2), trainingFunction, activationFunction);
    fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
    fprintf('Random Seed Used: %d\n', randomSeed);
    fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

 
end