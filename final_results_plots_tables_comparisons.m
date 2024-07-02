
% final_results_plots_tables_comparisons.m
clear; clc; clf; close all;

% Load data
load('./training-data/filtered_QHD_table.mat');
load('./training-data/filtered_QDP_table.mat');
load('./training-data/deleted_QHD_table.mat');
load('./training-data/deleted_QDP_table.mat');

% Extract data
QH = [filtered_QHD_table.FlowRate_m3h, filtered_QHD_table.Head_m]';
D = [filtered_QHD_table.Diameter_mm]';
QH_beps = [deleted_QHD_table.FlowRate_m3h, deleted_QHD_table.Head_m]';
D_beps = [deleted_QHD_table.Diameter_mm]';
QD = [filtered_QDP_table.FlowRate_m3h, filtered_QDP_table.Diameter_mm]';
P = [filtered_QDP_table.Power_kW]';
QD_beps = [deleted_QDP_table.FlowRate_m3h, deleted_QDP_table.Diameter_mm]';
P_beps = [deleted_QDP_table.Power_kW]';

% Create output directories
output_dir = 'final_results_plots_tables_comparisons';
figures_dir = fullfile(output_dir, 'figures');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% Hyperparameters based on latest optimization with GA
% randomSeed = 4837;
randomSeed = 4826;
nn_QHD_size_matrix = [ 15];
nn_QDH_size_matrix = [ 5,15];
nn_QDP_size_matrix = [2, 7, 29, 17];
maxEpochs = 355;
trainFcn = 'trainlm';


% Initialize results tables with headers
QHD_results = array2table(NaN(1, 8), 'VariableNames', {'DiameterRemoved', 'AvgMSE', ...
'TrainPerformance', 'ValPerformance', 'TestPerformance', 'MSEDeletedDiameter', 'MSEBEPs','Score'});
QDP_results = array2table(NaN(1, 8), 'VariableNames', {'DiameterRemoved', 'AvgMSE', ...
'TrainPerformance', 'ValPerformance', 'TestPerformance', 'MSEDeletedDiameter', 'MSEBEPs','Score'});
QDH_results = array2table(NaN(1, 8), 'VariableNames', {'DiameterRemoved', 'AvgMSE', ...
'TrainPerformance', 'ValPerformance', 'TestPerformance', 'MSEDeletedDiameter', 'MSEBEPs','Score'});

logs = {};  % Initialize logs

% Weights for different errors
weights = struct('train', 0, 'val', 0.05, 'test', 0.35, 'deleted_diameter', 0.40, 'beps', 0.20);

% Initialize best neural network variables for each type
bestNetQHD = struct('net', [], 'diameter', [], 'score', Inf, 'trainPerformance', [], ...
'valPerformance', [], 'testPerformance', []);
bestNetQDP = struct('net', [], 'diameter', [], 'score', Inf, 'trainPerformance', [], ...
'valPerformance', [], 'testPerformance', []);
bestNetQDH = struct('net', [], 'diameter', [], 'score', Inf, 'trainPerformance', [],...
 'valPerformance', [], 'testPerformance', []);






% Train on full dataset without any complete diameter removed on the best
% eff points are removed.
[trainedNetQHD, avgMSEsQHD, trainPerformanceQHD, valPerformanceQHD, testPerformanceQHD] =...
 train_nn(nn_QHD_size_matrix, maxEpochs, trainFcn, QH, D, randomSeed);
[trainedNetQDH, avgMSEsQDH, trainPerformanceQDH, valPerformanceQDH, testPerformanceQDH] = ...
train_nn(nn_QDH_size_matrix, maxEpochs, trainFcn, [QH(1,:); D], QH(2,:), randomSeed);
[trainedNetQDP, avgMSEsQDP, trainPerformanceQDP, valPerformanceQDP, testPerformanceQDP] = ...
train_nn(nn_QDP_size_matrix, maxEpochs, trainFcn, QD, P, randomSeed);

% Calculate mse_beps for the trained networks
mse_beps_QHD = perform(trainedNetQHD, D_beps, trainedNetQHD(QH_beps));

% Compute the weighted score
score_QHD = compute_score(trainPerformanceQHD, valPerformanceQHD, ...
testPerformanceQHD, NaN, mse_beps_QHD, weights);
QHD_results = [QHD_results; {NaN, avgMSEsQHD, trainPerformanceQHD, ...
valPerformanceQHD, testPerformanceQHD, NaN, mse_beps_QHD,score_QHD}];



if score_QHD < bestNetQHD.score
bestNetQHD.net = trainedNetQHD;
bestNetQHD.diameter = NaN;
bestNetQHD.score = mse_beps_QHD;
bestNetQHD.trainPerformance = trainPerformanceQHD;
bestNetQHD.valPerformance = valPerformanceQHD;
bestNetQHD.testPerformance = testPerformanceQHD;
end
% Plot test data vs trained net predictions
figure;
 % Original data
plot(QH(1,:), D , 'bo', 'DisplayName', 'Original Data');
hold on;
% Trained net predictions
plot(QH(1,:), trainedNetQHD(QH), 'r*', 'DisplayName', 'Trained Net Predictions'); 
% BEPs data
plot(QH_beps(1,:), D_beps, 'ms', 'DisplayName', 'BEPs Data'); 
legend('Location', 'best');
title('QHD: No Diameter removed during training only beps ');
xlabel('Flow Rate (m^3/h)');
ylabel('Diameter (mm)');
xlim([0 400]);
ylim([0 300]);
grid on;
hold off;
saveas(gcf, fullfile(figures_dir, 'QHD_Diameter_NaN.png'));



mse_beps_QDH = perform(trainedNetQDH, QH_beps(2,:), trainedNetQDH([QH_beps(1,:); D_beps]));
% Compute the weighted score
score_QDH = compute_score(trainPerformanceQDH, valPerformanceQDH, testPerformanceQDH, ... 
NaN, mse_beps_QDH, weights);
% Update QDH_results

QDH_results = [QDH_results; {NaN, avgMSEsQDH, trainPerformanceQDH, valPerformanceQDH, ... 
testPerformanceQDH, NaN, mse_beps_QDH,score_QDH}];

if score_QDH < bestNetQDH.score
bestNetQDH.net = trainedNetQDH;
bestNetQDH.diameter = NaN;
bestNetQDH.score = score_QDH;
bestNetQDH.trainPerformance = trainPerformanceQDH;
bestNetQDH.valPerformance = valPerformanceQDH;
bestNetQDH.testPerformance = testPerformanceQDH;
end
% Plot test data vs trained net predictions
figure;
% Original data
plot(QH(1,:), QH(2,:), 'bo', 'DisplayName', 'Original Data'); 
hold on;
% Trained net predictions
plot(QH(1,:), trainedNetQDH([QH(1,:); D]), 'r*', 'DisplayName', 'Trained Net Predictions'); 
% BEPs data
plot(QH_beps(1,:), QH_beps(2,:), 'ms', 'DisplayName', 'BEPs Data'); 
legend('Location', 'best');
title(['QDH: Diameter NaN']);
xlabel('Flow Rate (m^3/h)');
ylabel('Head (m)');
xlim([0 400]);
ylim([0 90]);
grid on;
hold off;
saveas(gcf, fullfile(figures_dir, ['QDH_Diameter_NaN.png']));

mse_beps_QDP = perform(trainedNetQDP, P_beps, trainedNetQDP(QD_beps));
% Compute the weighted score
score_QDP = compute_score(trainPerformanceQDP, valPerformanceQDP, testPerformanceQDP,...
 NaN, mse_beps_QDP, weights);

% Update QDP_results
QDP_results = [QDP_results; {NaN, avgMSEsQDP, trainPerformanceQDP, valPerformanceQDP, ...
testPerformanceQDP, NaN, mse_beps_QDP,score_QDP}];

if score_QDP < bestNetQDP.score
bestNetQDP.net = trainedNetQDP;
bestNetQDP.diameter = NaN;
bestNetQDP.score = score_QDP;
bestNetQDP.trainPerformance = trainPerformanceQDP;
bestNetQDP.valPerformance = valPerformanceQDP;
bestNetQDP.testPerformance = testPerformanceQDP;
end

% Plot test data vs trained net predictions
figure;
% Original data
plot(QD(1,:), P, 'bo', 'DisplayName', 'Original Data'); 
hold on;
% Trained net predictions
plot(QD(1,:), trainedNetQDP(QD), 'r*', 'DisplayName', 'Trained Net Predictions'); 
% BEPs data
plot(QD_beps(1,:), P_beps, 'ko', 'MarkerFaceColor', 'g', 'DisplayName', 'BEPs Data'); 
legend('Location', 'best');
title(['QDP: Diameter NaN' ]);
xlabel('Flow Rate (m^3/h)');
ylabel('Power (kW)');
xlim([0 400]);
ylim([0 120]);
grid on;
hold off;
saveas(gcf, fullfile(figures_dir, ['QDP_Diameter_NaN.png']));


% Loop to train on different NN on a dataset where one whole diameter curve is  hidden for QHD dataset

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

    try
        [trainedNetQHD_temp, avgMSEsQHD_temp, trainPerformanceQHD_temp, valPerformanceQHD_temp,...
         testPerformanceQHD_temp] = train_nn(nn_QHD_size_matrix, maxEpochs, trainFcn, QH_temp, D_temp, randomSeed);
        mse_deleted_diameter = perform(trainedNetQHD_temp, removedD, trainedNetQHD_temp(removedQH));
        mse_beps = perform(trainedNetQHD_temp, D_beps, trainedNetQHD_temp(QH_beps));
        logs{end+1} = ['Trained nn_QHD_temp on dataset without diameter ' num2str(diameterToRemove) ' successfully.'];

        % Compute the weighted score
        score = compute_score(trainPerformanceQHD_temp, valPerformanceQHD_temp,...
         testPerformanceQHD_temp, mse_deleted_diameter, mse_beps, weights);

        % Update QHD_results
        QHD_results = [QHD_results; {diameterToRemove, avgMSEsQHD_temp, trainPerformanceQHD_temp,...
         valPerformanceQHD_temp, testPerformanceQHD_temp, mse_deleted_diameter, mse_beps,score}];

        % Check if this is the best neural network for QHD
        if score < bestNetQHD.score
            bestNetQHD.net = trainedNetQHD_temp;
            bestNetQHD.diameter = diameterToRemove;
            bestNetQHD.score = score;
            bestNetQHD.trainPerformance = trainPerformanceQHD_temp;
            bestNetQHD.valPerformance = valPerformanceQHD_temp;
            bestNetQHD.testPerformance = testPerformanceQHD_temp;
        end

        % Plot test data vs trained net predictions
        figure;
        % Original data
        plot(QH(1,:),D, 'bo', 'DisplayName', 'Original Data'); 
        hold on;
        % Trained net predictions
        plot(QH(1,:), trainedNetQHD_temp(QH), 'r*', 'DisplayName', 'Trained Net Predictions'); 
        % Removed diameter data
        plot(removedQH(1,:), removedD, 'gx', 'DisplayName', 'Removed Diameter Data'); 
        % BEPs data
        plot(QH_beps(1,:), D_beps, 'ms', 'DisplayName', 'BEPs Data'); 
        legend('Location', 'best');
        title(['QHD: Diameter ' num2str(diameterToRemove)]);
        xlabel('Flow Rate (m^3/h)');
        ylabel('Diameter (mm)');
        xlim([0 400]);
        ylim([0 300]);
        grid on;
        hold off;
        saveas(gcf, fullfile(figures_dir, ['QHD_Diameter_' num2str(diameterToRemove) '.png']));
        logs{end+1} = ['Plotted and saved QHD data for diameter ' num2str(diameterToRemove) ' successfully.'];
    catch e
        logs{end+1} = ['Error processing diameter ' num2str(diameterToRemove) ': ' e.message];
    end
end

% Loop to train on different NN on a dataset where one whole diameter curve is  hidden for QHD dataset
for dIdx = 1:length(distinctDiametersQHD)
    diameterToRemove = distinctDiametersQHD(dIdx);
    indicesToRemove = find(D == diameterToRemove);
    removedQH = QH(:, indicesToRemove);
    removedD = D(indicesToRemove);
    QH_temp = QH;
    D_temp = D;
    QH_temp(:, indicesToRemove) = [];
    D_temp(:, indicesToRemove) = [];

    try
        [trainedNetQDH_temp, avgMSEsQDH_temp, trainPerformanceQDH_temp, valPerformanceQDH_temp,...
         testPerformanceQDH_temp] = train_nn(nn_QDH_size_matrix, maxEpochs, trainFcn, ...
         [QH_temp(1,:); D_temp], QH_temp(2,:), randomSeed);
        mse_deleted_diameter = perform(trainedNetQDH_temp, removedQH(2,:),...
         trainedNetQDH_temp([removedQH(1,:); removedD]));
        mse_beps = perform(trainedNetQDH_temp, QH_beps(2,:), trainedNetQDH_temp([QH_beps(1,:); D_beps]));
        logs{end+1} = ['Trained nn_QDH_temp on dataset without diameter ' num2str(diameterToRemove) ' successfully.'];

        % Compute the weighted score
        score = compute_score(trainPerformanceQDH_temp, valPerformanceQDH_temp, ...
        testPerformanceQDH_temp, mse_deleted_diameter, mse_beps, weights);

        % Update QDH_results
        QDH_results = [QDH_results; {diameterToRemove, avgMSEsQDH_temp, trainPerformanceQDH_temp,...
         valPerformanceQDH_temp, testPerformanceQDH_temp, mse_deleted_diameter, mse_beps,score}];

        % Check if this is the best neural network for QDH
        if score < bestNetQDH.score
            bestNetQDH.net = trainedNetQDH_temp;
            bestNetQDH.diameter = diameterToRemove;
            bestNetQDH.score = score;
            bestNetQDH.trainPerformance = trainPerformanceQDH_temp;
            bestNetQDH.valPerformance = valPerformanceQDH_temp;
            bestNetQDH.testPerformance = testPerformanceQDH_temp;
        end

        % Plot test data vs trained net predictions
        figure;
        % Original data
        plot(QH(1,:), QH(2,:), 'bo', 'DisplayName', 'Original Data'); 
        hold on;
        % Trained net predictions
        plot(QH(1,:), trainedNetQDH_temp([QH(1,:); D]), 'r*', 'DisplayName', 'Trained Net Predictions'); 
         % Removed diameter data
        plot(removedQH(1,:), removedQH(2,:), 'gx', 'DisplayName', 'Removed Diameter Data');
        % BEPs data
        plot(QH_beps(1,:), QH_beps(2,:), 'ms', 'DisplayName', 'BEPs Data'); 
        legend('Location', 'best');
        title(['QDH: Diameter ' num2str(diameterToRemove)]);
        xlabel('Flow Rate (m^3/h)');
        ylabel('Head (m)');
        xlim([0 400]);
        ylim([0 90]);
        grid on;
        hold off;
        saveas(gcf, fullfile(figures_dir, ['QDH_Diameter_' num2str(diameterToRemove) '.png']));
        logs{end+1} = ['Plotted and saved QDH data for diameter ' num2str(diameterToRemove) ' successfully.'];
    catch e
        logs{end+1} = ['Error processing diameter ' num2str(diameterToRemove) ': ' e.message];
    end
end

% Loop to train on different NN on a dataset where one whole diameter curve is  hidden for QDP dataset
distinctDiametersQDP = unique(QD(2,:));
for dIdx = 1:length(distinctDiametersQDP)
    diameterToRemove = distinctDiametersQDP(dIdx);
    indicesToRemove = find(QD(2,:) == diameterToRemove);
    removedQD = QD(:, indicesToRemove);
    removedP = P(indicesToRemove);
    QD_temp = QD;
    P_temp = P;
    QD_temp(:, indicesToRemove) = [];
    P_temp(:, indicesToRemove) = [];

    try
        [trainedNetQDP_temp, avgMSEsQDP_temp, trainPerformanceQDP_temp,...
         valPerformanceQDP_temp, testPerformanceQDP_temp] = train_nn(nn_QDP_size_matrix, maxEpochs,...
          trainFcn, QD_temp, P_temp, randomSeed);
        mse_deleted_diameter = perform(trainedNetQDP_temp, removedP, trainedNetQDP_temp(removedQD));
        mse_beps = perform(trainedNetQDP_temp, P_beps, trainedNetQDP_temp(QD_beps));
        logs{end+1} = ['Trained nn_QDP_temp on dataset without diameter ' num2str(diameterToRemove) ' successfully.'];

        % Compute the weighted score
        score = compute_score(trainPerformanceQDP_temp, valPerformanceQDP_temp, testPerformanceQDP_temp,...
         mse_deleted_diameter, mse_beps, weights);

        % Update QDP_results
        QDP_results = [QDP_results; {diameterToRemove, avgMSEsQDP_temp, trainPerformanceQDP_temp,...
         valPerformanceQDP_temp, testPerformanceQDP_temp, mse_deleted_diameter, mse_beps,score}];

        % Check if this is the best neural network for QDP
        if score < bestNetQDP.score
            bestNetQDP.net = trainedNetQDP_temp;
            bestNetQDP.diameter = diameterToRemove;
            bestNetQDP.score = score;
            bestNetQDP.trainPerformance = trainPerformanceQDP_temp;
            bestNetQDP.valPerformance = valPerformanceQDP_temp;
            bestNetQDP.testPerformance = testPerformanceQDP_temp;
        end

        % Plot test data vs trained net predictions
        figure;
        % Original data
        plot(QD(1,:), P, 'bo', 'DisplayName', 'Original Data'); 
        hold on;
        % Trained net predictions
        plot(QD(1,:), trainedNetQDP_temp(QD), 'r*', 'DisplayName', 'Trained Net Predictions');
        % Removed diameter data 
        plot(removedQD(1,:), removedP, 'gx', 'DisplayName', 'Removed Diameter Data'); 
        % BEPs data
        plot(QD_beps(1,:), P_beps, 'ko', 'MarkerFaceColor', 'g', 'DisplayName', 'BEPs Data'); 
        legend('Location', 'best');
        title(['QDP: Diameter ' num2str(diameterToRemove)]);
        xlabel('Flow Rate (m^3/h)');
        ylabel('Power (kW)');
        xlim([0 400]);
        ylim([0 120]);
        grid on;
        hold off;
        saveas(gcf, fullfile(figures_dir, ['QDP_Diameter_' num2str(diameterToRemove) '.png']));
        logs{end+1} = ['Plotted and saved QDP data for diameter ' num2str(diameterToRemove) ' successfully.'];
    catch e
        logs{end+1} = ['Error processing diameter ' num2str(diameterToRemove) ': ' e.message];
    end
end

% Remove the initial row of NaN values from the results tables
QHD_results(1, :) = [];
QDP_results(1, :) = [];
QDH_results(1, :) = [];

% Save the results and logs
writetable(QHD_results, fullfile(output_dir, 'QHD_results.csv'));
writetable(QDP_results, fullfile(output_dir, 'QDP_results.csv'));
writetable(QDH_results, fullfile(output_dir, 'QDH_results.csv'));
writecell(logs, fullfile(output_dir, 'logs.txt'));

% Save the best neural networks
if isfield(bestNetQHD, 'net')
    save(fullfile(output_dir, 'bestNetQHD.mat'), 'bestNetQHD');
end
if isfield(bestNetQDP, 'net')
    save(fullfile(output_dir, 'bestNetQDP.mat'), 'bestNetQDP');
end
if isfield(bestNetQDH, 'net')
    save(fullfile(output_dir, 'bestNetQDH.mat'), 'bestNetQDH');
end

% Display the results
disp('QHD Results:');
disp(QHD_results);
disp('QDP Results:');
disp(QDP_results);
disp('QDH Results:');
disp(QDH_results);

% Display the best neural networks and their corresponding diameters
disp('Best Neural Networks:');
if isfield(bestNetQHD, 'net')
    disp(['Best QHD Network: Diameter ' num2str(bestNetQHD.diameter)]);
    disp(['Score: ' num2str(bestNetQHD.score)]);
    disp(['Train Performance: ' num2str(bestNetQHD.trainPerformance)]);
    disp(['Validation Performance: ' num2str(bestNetQHD.valPerformance)]);
    disp(['Test Performance: ' num2str(bestNetQHD.testPerformance)]);
end
if isfield(bestNetQDP, 'net')
    disp(['Best QDP Network: Diameter ' num2str(bestNetQDP.diameter)]);
    disp(['Score: ' num2str(bestNetQDP.score)]);
    disp(['Train Performance: ' num2str(bestNetQDP.trainPerformance)]);
    disp(['Validation Performance: ' num2str(bestNetQDP.valPerformance)]);
    disp(['Test Performance: ' num2str(bestNetQDP.testPerformance)]);
end
if isfield(bestNetQDH, 'net')
    disp(['Best QDH Network: Diameter ' num2str(bestNetQDH.diameter)]);
    disp(['Score: ' num2str(bestNetQDH.score)]);
    disp(['Train Performance: ' num2str(bestNetQDH.trainPerformance)]);
    disp(['Validation Performance: ' num2str(bestNetQDH.valPerformance)]);
    disp(['Test Performance: ' num2str(bestNetQDH.testPerformance)]);
end

%%  compare to nn to traditional methods 
distinctDiameters = unique(D);


% Extract the Q, H and D is already there in D.
Q = QH(1,:);
H = QH(2,:);

% Create a structure to hold Q,H curves for each diameter in unique_D.
pump_data = struct('Diameter', cell(length(distinctDiameters), 1), 'Q',...
 cell(length(distinctDiameters), 1), 'H', cell(length(distinctDiameters), 1));

for i = 1:length(distinctDiameters)
    idx = (D == distinctDiameters(i));
    pump_data(i).Diameter = distinctDiameters(i);
    pump_data(i).Q = Q(idx);
    pump_data(i).H = H(idx);
end

% Initialize arrays to store errors and diameter reductions
percent_errors_cas_260 = zeros(1, 5);
percent_errors_cas_nearest = zeros(1, 5);
percent_errors_nn = zeros(1, 5);
percent_reductions = zeros(1, 5);

% Loop through each column in QH_beps
for i = 1:4
    d_real = D_beps(1, i); % Extracting d_real from D_beps

    % Calculate d using constant_area_scaling
    d_trimmed_cas_260 = constant_area_scaling(QH_beps(1, i),...
     QH_beps(2,i), pump_data(5).H,pump_data(5).Q,  pump_data(5).Diameter, 4);
    percent_errors_cas_260(i) = abs((d_trimmed_cas_260 - d_real) / d_real) * 100;

    
    % Calculate d using trainedNetQHD
    d_trimmed_nn = bestNetQHD.net(QH_beps(:, i));
    percent_errors_nn(i) = abs((d_trimmed_nn - d_real) / d_real) * 100;


end

% Store the errors and percent reductions in CSV files
errors_table = table((1:5)', percent_errors_cas_260',  percent_errors_nn',  ...
    'VariableNames', {'Index', '%error  (constand area scaling)',  '% error in (neural network)'});

writetable(errors_table, fullfile(output_dir, 'percent_error_between_bestQ_HD_NN_CAS.csv'));


% Display the percent errors
disp('Percent errors for traditional trimming when the 260 mm diameter is ref:');
disp(percent_errors_cas_260);


disp('Percent errors for best trainedNetQHD:');
disp(percent_errors_nn);



% create and save our first time style 3d plot.
processDataAndVisualize(QH', D', QD',P', bestNetQHD.net, bestNetQDP.net,figures_dir);
% Display a message indicating completion
disp('Script execution completed.');




%% functions

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Function to compute the weighted score, omitting NaN values
function score = compute_score(trainPerf, valPerf, testPerf, mseDeleted, mseBEPS, weights) 
if ~isnan(mseDeleted)
        score = weights.train * trainPerf + ...
            weights.val * valPerf + ...
            weights.test * testPerf + ...
            weights.deleted_diameter * mseDeleted + weights.beps * mseBEPS;
else
       score = weights.train * trainPerf + ...
            weights.val * valPerf + ...
            weights.test * testPerf  + weights.beps * mseBEPS;
end
end

function [trainedNet,avgMSEs,trainPerformance,valPerformance,testPerformance] = ... 
    train_nn(nn_size_matrix,maxEpochs,trainFcn ,x, t, randomSeed)
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
        trainPerformance    = perform(trainedNet,trainTargets,predictions);
        valPerformance      = perform(trainedNet,valTargets,predictions);
        testPerformance     = perform(trainedNet,testTargets,predictions);

        % for better performance we came up with this convention rather than
        % using the mse based on perform function only
        avgMSEs = (mse +  ...
            trainPerformance +...
            valPerformance+....
            testPerformance) / 4;
        
        
end







function D2_prime = constant_area_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree)

    p = polyfit(Q_curve, H_curve, poly_degree);
    
    % Calculate the scaled A value
    A = H_prime / (Q_prime)^2;

    syms Q
    poly_expr = poly2sym(p, Q);
    
    eqn = A * Q^2 == poly_expr;
    sol = double(solve(eqn, Q));
    
    Q_valid = sol(sol > 0 & imag(sol) == 0 & sol <= max(Q_curve) & sol >= min(Q_curve));
    if isempty(Q_valid)
        error('No valid intersection found within the range of the pump curve.');
    end
    Q_intersect = max(Q_valid);
    
    D2_prime = Q_prime / Q_intersect * D2;
end

function error = fitness_polyfit(degree, Q, H)
    degree = round(degree); % Ensure the degree is an integer
    
    % Center and scale the data
    Q_mean = mean(Q);
    Q_std = std(Q);
    H_mean = mean(H);
    H_std = std(H);
    Q_scaled = (Q - Q_mean) / Q_std;
    H_scaled = (H - H_mean) / H_std;
    
    p = polyfit(Q_scaled, H_scaled, degree);
    H_fit = polyval(p, Q_scaled) * H_std + H_mean;
    error = norm(H - H_fit) / norm(H);
end


function processDataAndVisualize(QH, D, QD, P, bestTrainedNetD, bestTrainedNetP,figures_dir,saveFigures)
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


    saveas(gcf,fullfile(figures_dir,['best_nn_diameter_power_visualization_' char(currentTime) '.fig']));
    saveas(gcf,fullfile(figures_dir,['best_nn_diameter_power_visualization_' char(currentTime) '.png']));
end
end
