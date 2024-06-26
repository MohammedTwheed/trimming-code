clear; clc; clf; close all;
load('filtered_QHD_table.mat')
load('filtered_QDP_table.mat')
load('deleted_QHD_table.mat')
load('deleted_QDP_table.mat')

QH = [filtered_QHD_table.FlowRate_m3h, filtered_QHD_table.Head_m]';
D  = [filtered_QHD_table.Diameter_mm]';

QH_beps = [deleted_QHD_table.FlowRate_m3h, deleted_QHD_table.Head_m]';
D_beps = [deleted_QHD_table.Diameter_mm]';

QD = [filtered_QDP_table.FlowRate_m3h, filtered_QDP_table.Diameter_mm]';
P = [filtered_QDP_table.Power_kW]';

QD_beps = [deleted_QDP_table.FlowRate_m3h, deleted_QDP_table.Diameter_mm]';
P_beps = [deleted_QDP_table.Power_kW]';

% these hyper params are based on our latest optimization with ga
% now we want to finalize our work.

randomSeed = 4837;
nn_QHD_size_matrix = [2,16];
nn_QDH_size_matrix = [2,16];
nn_QDP_size_matrix = [2,7,29,17];
maxEpochs = 191;
trainFcn= 'trainlm';

% Train on full dataset
[trainedNetQHD, avgMSEsQHD, trainPerformanceQHD, valPerformanceQHD, testPerformanceQHD] = train_nn(nn_QHD_size_matrix, maxEpochs, trainFcn, QH, D, randomSeed);
[trainedNetQDH, avgMSEsQDH, trainPerformanceQDH, valPerformanceQDH, testPerformanceQDH] = train_nn(nn_QDH_size_matrix, maxEpochs, trainFcn, [QH(1,:); D], QH(2,:), randomSeed);
[trainedNetQDP, avgMSEsQDP, trainPerformanceQDP, valPerformanceQDP, testPerformanceQDP] = train_nn(nn_QDP_size_matrix, maxEpochs, trainFcn, QD, P, randomSeed);

% Arrays to save performance metrics
QHD_results = [];
QDP_results = [];

%% loop to train on different diameters hidden for QHD dataset
distinctDiametersQHD = unique(D);
for dIdx = 1:length(distinctDiametersQHD)
    diameterToRemove = distinctDiametersQHD(dIdx);
    indicesToRemove = find(D == diameterToRemove);
    removedQH = QH(:, indicesToRemove);
    removedD = D(indicesToRemove);
    QH_temp = QH;
    D_temp = D;
    QH_temp(:, indicesToRemove) = [];
    D_temp(:, indicesToRemove) = [];

    [trainedNetQHD_temp, avgMSEsQHD_temp, trainPerformanceQHD_temp, valPerformanceQHD_temp, testPerformanceQHD_temp] = train_nn(nn_QHD_size_matrix, maxEpochs, trainFcn, QH_temp, D_temp, randomSeed);
    mse_deleted_diameter = perform(trainedNetQHD_temp, removedD, trainedNetQHD_temp(removedQH));
    mse_beps = perform(trainedNetQHD_temp, D_beps, trainedNetQHD_temp(QH_beps));


    [trainedNetQDH_temp, avgMSEsQDH_temp, trainPerformanceQDH_temp, valPerformanceQDH_temp, testPerformanceQDH_temp] = train_nn(nn_QDH_size_matrix, maxEpochs, trainFcn, [QH_temp(1,:); D_temp], QH_temp(2,:), randomSeed);

    QHD_results = [QHD_results; diameterToRemove, avgMSEsQHD_temp, trainPerformanceQHD_temp, valPerformanceQHD_temp, testPerformanceQHD_temp, mse_deleted_diameter, mse_beps];

    % Plot test data vs trained net predictions
    figure;
    plot(QH(1,:), QH(2,:), 'bo', 'DisplayName', 'Original Data'); % Original data
    hold on;
    plot(QH_temp(1,:), trainedNetQDH_temp([QH_temp(1,:); D_temp]), 'r*', 'DisplayName', 'Trained Net Predictions'); % Trained net predictions
    plot(removedQH(1,:), removedQH(2,:), 'gx', 'DisplayName', 'Removed Diameter Data'); % Removed diameter data
    plot(QH_beps(1,:), QH_beps(2,:), 'ms', 'DisplayName', 'BEPs Data'); % BEPs data
    legend('Location', 'best');
    title(['QHD: Diameter ' num2str(diameterToRemove)]);
    xlabel('Flow Rate (m^3/h)');
    ylabel('Head (m)');
    xlim([0 400]);
    ylim([0 90]);
    grid on;
    hold off;
end

%% loop to train on different diameters hidden for QDP dataset
distinctDiametersQDP = unique(QD(2,:));
for dIdx = 1:length(distinctDiametersQDP)
    diameterToRemove = distinctDiametersQDP(dIdx);
    indicesToRemove = find(QD(2,:) == diameterToRemove);
    removedQD = QD(:, indicesToRemove);
    removedP = P(indicesToRemove);
    QD_temp = QD;
    P_temp = P;
    QD_temp(:, indicesToRemove) = [];
    P_temp(indicesToRemove) = [];

    [trainedNetQDP_temp, avgMSEsQDP_temp, trainPerformanceQDP_temp, valPerformanceQDP_temp, testPerformanceQDP_temp] = train_nn(nn_QDP_size_matrix, maxEpochs, trainFcn, QD_temp, P_temp, randomSeed);%[2,7,29,17]
    mse_deleted_diameter = perform(trainedNetQDP_temp, removedP, trainedNetQDP_temp(removedQD));
    mse_beps = perform(trainedNetQDP_temp, P_beps, trainedNetQDP_temp(QD_beps));

    QDP_results = [QDP_results; diameterToRemove, avgMSEsQDP_temp, trainPerformanceQDP_temp, valPerformanceQDP_temp, testPerformanceQDP_temp, mse_deleted_diameter, mse_beps];

    % Plot test data vs trained net predictions
    figure;
    plot(QD(1,:), P, 'bo', 'DisplayName', 'Original Data'); % Original data
    hold on;
    plot(QD(1,:), trainedNetQDP_temp(QD), 'r*', 'DisplayName', 'Trained net predictions');
    plot(removedQD(1,:), removedP, 'gx', 'DisplayName', 'Removed Diameter Data'); % Removed diameter data
    plot(QD_beps(1,:), P_beps, 'ms', 'DisplayName', 'BEPs Data'); % BEPs data
    legend('Location', 'best');
    title(['QDP: Diameter ' num2str(diameterToRemove)]);
    xlabel('Flow Rate (m^3/h)');
    ylabel('Power (kW)');
    xlim([0 400]);
    ylim([0 90]);
    grid on;
    hold off;
end

% Save results to CSV
QHD_results_table = array2table(QHD_results, 'VariableNames', {'Diameter', 'AvgMSEs', 'TrainPerformance', 'ValPerformance', 'TestPerformance', 'MSE_Deleted_Diameter', 'MSE_BEPS'});
QDP_results_table = array2table(QDP_results, 'VariableNames', {'Diameter', 'AvgMSEs', 'TrainPerformance', 'ValPerformance', 'TestPerformance', 'MSE_Deleted_Diameter', 'MSE_BEPS'});

writetable(QHD_results_table, 'QHD_results.csv');
writetable(QDP_results_table, 'QDP_results.csv');

% Display results in a table format
disp('QHD Results:');
disp(QHD_results_table);

disp('QDP Results:');
disp(QDP_results_table);

%% functions

function [trainedNet, avgMSEs, trainPerformance, valPerformance, testPerformance] = train_nn(nn_size_matrix, maxEpochs, trainFcn, x, t, randomSeed)
    rng(randomSeed); % Set random seed for reproducibility.
    net = feedforwardnet(nn_size_matrix, trainFcn);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = maxEpochs;
    net.performFcn = 'mse';
    net.input.processFcns = {'removeconstantrows', 'mapminmax'};
    net.output.processFcns = {'removeconstantrows', 'mapminmax'};
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio = 0.15;
    net.divideParam.testRatio = 0.15;

    [trainedNet, tr] = train(net, x, t);
    predictions = trainedNet(x);
    mse = perform(trainedNet, t, predictions);

    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net, trainTargets, predictions);
    valPerformance = perform(net, valTargets, predictions);
    testPerformance = perform(net, testTargets, predictions);

    avgMSEs = (mse + trainPerformance + valPerformance + testPerformance) / 4;
end
