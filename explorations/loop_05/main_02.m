%% 
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

% these hyper params are based on our latest optimization with ga
% now we want to finalize our work.

randomSeed = 4837;
nn_QHD_size_matrix = [2,16];
nn_QDH_size_matrix = [2,16];
nn_QDP_size_matrix = [12,15];
maxEpochs = 191;
trainFcn= 'trainlm';

[trainedNetQHD,avgMSEsQHD,trainPerformanceQHD,valPerformanceQHD,testPerformanceQHD] = train_nn([2,15,],maxEpochs,trainFcn ,QH, D, randomSeed);
[trainedNetQDH,avgMSEsQDH,trainPerformanceQDH,valPerformanceQDH,testPerformanceQDH] = train_nn([2,15,],maxEpochs,trainFcn ,[QH(1,:); D], QH(2,:), randomSeed);
[trainedNetQDP,avgMSEsQDP,trainPerformanceQDP,valPerformanceQDP,testPerformanceQDP] = train_nn([12,15,],maxEpochs,trainFcn ,QD, P, randomSeed);

%% loop to train on different diameters hidden for QHD dataset

% here is distinctDiameters based on QHD dataset.
distinctDiametersQHD = unique(D);


for dIdx = 1:length(distinctDiametersQHD)
    % Current diameter to remove
    diameterToRemove = distinctDiametersQHD(dIdx);
    
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
    

[trainedNetQHD,avgMSEsQHD,trainPerformanceQHD,valPerformanceQHD,testPerformanceQHD] = train_nn([2,15,],maxEpochs,trainFcn ,QH_temp, D_temp, randomSeed);
[trainedNetQDH,avgMSEsQDH,trainPerformanceQDH,valPerformanceQDH,testPerformanceQDH] = train_nn([2,15,],maxEpochs,trainFcn ,[QH_temp(1,:); D_temp], QH_temp(2,:), randomSeed);



end



%% loop to train on different diameters hidden for QDP dataset

% Get unique diameters for the QDP dataset
distinctDiametersQDP = unique(QD(2,:));

% Extract the Q and D is already there in QD.
Q_QDP = QD(1,:);
D_QDP = QD(2,:);

% Create a structure to hold Q,P curves for each diameter in QDP dataset.
pump_data_QDP = struct('Diameter', cell(length(distinctDiametersQDP), 1), 'Q', cell(length(distinctDiametersQDP), 1), 'P', cell(length(distinctDiametersQDP), 1));

for i = 1:length(distinctDiametersQDP)
    idx = (D_QDP == distinctDiametersQDP(i));
    pump_data_QDP(i).Diameter = distinctDiametersQDP(i);
    pump_data_QDP(i).Q = Q_QDP(idx);
    pump_data_QDP(i).P = P(idx);
end

for dIdx = 1:length(distinctDiametersQDP)
    % Current diameter to remove
    diameterToRemove = distinctDiametersQDP(dIdx);
    
    % Find indices of the current diameter in D_QDP
    indicesToRemove = find(D_QDP == diameterToRemove);
    
    % Store the removed data for later use
    removedQD = QD(:, indicesToRemove);
    removedP = P(indicesToRemove);
    
    % Remove rows from QD and P based on the indices
    QD_temp = QD;
    P_temp = P;
    QD_temp(:, indicesToRemove) = [];
    P_temp(indicesToRemove) = [];
    
    [trainedNetQDP, avgMSEsQDP, trainPerformanceQDP, valPerformanceQDP, testPerformanceQDP] = train_nn([12,15,], maxEpochs, trainFcn, QD_temp, P_temp, randomSeed);
end


%% functions


function [trainedNet,avgMSEs,trainPerformance,valPerformance,testPerformance] = train_nn(nn_size_matrix,maxEpochs,trainFcn ,x, t, randomSeed)
        rng(randomSeed); % Set random seed for reproducibility.
        % Define the neural network.
        net = feedforwardnet(nn_size_matrix,trainFcn);
        % Suppress training GUI for efficiency.
        net.trainParam.showWindow = false;
        net.trainParam.epochs = maxEpochs;

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
        
        
end

