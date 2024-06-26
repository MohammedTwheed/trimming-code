

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
%%



% these hyper params are based on our latest optimization with ga
% now we want to finalize our work.

randomSeed = 4837;
nn_size_matrix = [2,16];
maxEpochs = 191;




trainFcn= 'trainlm';


[trainedNetQHD,avgMSEsQHD,trainPerformanceQHD,valPerformanceQHD,testPerformanceQHD] = train_nn([2,15,],maxEpochs,trainFcn ,QH, D, randomSeed);
[trainedNetQDH,avgMSEsQDH,trainPerformanceQDH,valPerformanceQDH,testPerformanceQDH] = train_nn([2,15,],maxEpochs,trainFcn ,[QH(1,:); D], QH(2,:), randomSeed);
[trainedNetQDP,avgMSEsQDP,trainPerformanceQDP,valPerformanceQDP,testPerformanceQDP] = train_nn([12,15,],maxEpochs,trainFcn ,QD, P, randomSeed);

D2_prime = constant_width_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree);

processDataAndVisualize(QH', D', QD',P', trainedNetQHD, trainedNetQDP);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% work


%% Find all distinct diameters in D
distinctDiameters = unique(D);


% Extract the Q, H and D is already there in D.
Q = QH(1,:);
H = QH(2,:);


% Create a structure to hold Q,H curves for each diameter in unique_D.
pump_data = struct('Diameter', cell(length(distinctDiameters), 1), 'Q', cell(length(distinctDiameters), 1), 'H', cell(length(distinctDiameters), 1));

for i = 1:length(distinctDiameters)
    idx = (D == distinctDiameters(i));
    pump_data(i).Diameter = distinctDiameters(i);
    pump_data(i).Q = Q(idx);
    pump_data(i).H = H(idx);
end

d_real                 = D_beps(1)
d_trimmed_cas_260      = constant_area_scaling(QH_beps(1,1),QH_beps(2,1),pump_data(5).Q,pump_data(5).H,pump_data(5).Diameter,4)
d_trimmed_cas_nearest  = trim_diameters(QH_beps(:,1),'filtered_QHD_table.mat')
d_trimmed_nn           = trainedNetQHD(QH_beps(:,1))

%%
% here i already know the length of beps is 5 for 5 diameters in my
% dataset.
% Initialize an array to store percent errors
percent_errors_cas_260 = zeros(1, 5);
percent_errors_cas_nearest = zeros(1, 5);
percent_errors_nn = zeros(1, 5);

% Loop through each column in QH_beps
for i = 1:5
    d_real = D_beps(1, i); % Extracting d_real from D_beps

    % Calculate d using constant_area_scaling 260
    d_trimmed_cas_260 = constant_area_scaling(QH_beps(1, i), QH_beps(2, i), pump_data(5).Q, pump_data(5).H, pump_data(5).Diameter, 4);
    percent_errors_cas_260(i) = abs((d_trimmed_cas_260 - d_real) / d_real) * 100;

    % Calculate d using trim_diameters
    d_trimmed_cas_nearest = trim_diameters(QH_beps(:, i), 'filtered_QHD_table.mat');
    percent_errors_cas_nearest(i) = abs((d_trimmed_cas_nearest - d_real) / d_real) * 100;

    % Calculate d using trainedNetQHD
    d_trimmed_nn = trainedNetQHD(QH_beps(:, i));
    percent_errors_nn(i) = abs((d_trimmed_nn - d_real) / d_real) * 100;
end

% Display the percent errors
disp('Percent errors for constant_area_scaling 260:');
disp(percent_errors_cas_260);

disp('Percent errors for trim_diameters:');
disp(percent_errors_cas_nearest);

disp('Percent errors for trainedNetQHD:');
disp(percent_errors_nn);


%% 

QH_beps = removedQH;

% Generate a 1x101 matrix of random real numbers between 3 and 7
random_values = 0.02 + (0.3-0.02) * rand(1, 101);

% Subtract the random values from the second row of QH
QH_beps(2, :) = QH_beps(2, :) - random_values;


D_beps = removedD;


% Initialize an array to store percent errors
percent_errors_cas_260 = zeros(1, 5);
percent_errors_cas_nearest = zeros(1, 5);
percent_errors_nn = zeros(1, 5);

% Loop through each column in QH_beps
for i = 1:5
    d_real = D_beps(1, i); % Extracting d_real from D_beps

    % Calculate d using constant_area_scaling
    d_trimmed_cas_260 = constant_area_scaling(QH_beps(1, i), QH_beps(2, i), pump_data(5).Q, pump_data(5).H, pump_data(5).Diameter, 4);
    percent_errors_cas_260(i) = abs((d_trimmed_cas_260 - d_real) / d_real) * 100;

    % Calculate d using trim_diameters
    d_trimmed_cas_nearest = trim_diameters(QH_beps(:, i), 'filtered_QHD_table.mat');
    percent_errors_cas_nearest(i) = abs((d_trimmed_cas_nearest - d_real) / d_real) * 100;

    % Calculate d using trainedNetQHD
    d_trimmed_nn = trainedNetQHD(QH_beps(:, i));
    percent_errors_nn(i) = abs((d_trimmed_nn - d_real) / d_real) * 100;
end

% Display the percent errors
disp('Percent errors for constant_area_scaling:');
disp(percent_errors_cas_260);

disp('Percent errors for trim_diameters:');
disp(percent_errors_cas_nearest);

disp('Percent errors for trainedNetQHD:');
disp(percent_errors_nn);

% Calculate the Mean Absolute Error (MAE)
mae_trim_diameters = mean(percent_errors_cas_nearest);
mae_trainedNetQHD = mean(percent_errors_nn);

% Compare the MAEs
if mae_trainedNetQHD < mae_trim_diameters
    disp('trainedNetQHD has a lower mean absolute error and is therefore better.');
else
    disp('trim_diameters has a lower mean absolute error and is therefore better.');
end

% Count how many times one method outperforms the other
count_better_trainedNetQHD = sum(percent_errors_nn < percent_errors_cas_nearest);
count_better_trim_diameters = sum(percent_errors_cas_nearest < percent_errors_nn);

if count_better_trainedNetQHD > count_better_trim_diameters
    disp('trainedNetQHD outperforms trim_diameters more frequently.');
else
    disp('trim_diameters outperforms trainedNetQHD more frequently.');
end

%%

% for dIdx = 1:length(distinctDiameters)
    % Current diameter to remove
    diameterToRemove = distinctDiameters(3);
    
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




% end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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






function D_trimmed = trim_diameters(QH, dataPath)

    % Load the data
    loadedData = load(dataPath);
    filtered_QHD_table = loadedData.filtered_QHD_table;

    % Extract columns
    Q = filtered_QHD_table.FlowRate_m3h;
    H = filtered_QHD_table.Head_m;
    D = filtered_QHD_table.Diameter_mm;

    % Get unique diameters
    unique_D = unique(D);

    % Initialize pump_data structure
    pump_data = struct('Diameter', cell(length(unique_D), 1), 'Q', cell(length(unique_D), 1), 'H', cell(length(unique_D), 1), 'BestDegree', cell(length(unique_D), 1));

    % Fill pump_data structure
    for i = 1:length(unique_D)
        idx = (D == unique_D(i));
        pump_data(i).Diameter = unique_D(i);
        pump_data(i).Q = Q(idx);
        pump_data(i).H = H(idx);
    end

    % Curve Fitting using Genetic Algorithm
    ga_options = optimoptions('ga', ...
        'Display', 'off', ...
        'MaxGenerations', 200, ...
        'MaxStallGenerations', 20, ...
        'FunctionTolerance', 1e-6, ...
        'PopulationSize', 50);

    for i = 1:length(pump_data)
        Q = pump_data(i).Q;
        H = pump_data(i).H;

        if isempty(Q) || isempty(H) || any(isnan(Q)) || any(isnan(H))
            disp(['Error: Empty or NaN data for dataset ', num2str(i)]);
            continue;
        end

        Q = Q(:);
        H = H(:);

        if length(Q) ~= length(H)
            disp(['Error: Mismatch in length of Q and H for dataset ', num2str(i)]);
            continue;
        end

        fitness_function = @(degree) fitness_polyfit(degree, Q, H);

        try
            [best_degree, ~] = ga(fitness_function, 1, [], [], [], [], 2, 20, [], ga_options);
        catch ga_error
            disp(['Error optimizing for dataset ', num2str(i)]);
            disp(ga_error.message);
            continue;
        end

        best_degree = max(1, round(best_degree));

        try
            % Center and scale the data
            Q_mean = mean(Q);
            Q_std = std(Q);
            H_mean = mean(H);
            H_std = std(H);
            Q_scaled = (Q - Q_mean) / Q_std;
            H_scaled = (H - H_mean) / H_std;

            p = polyfit(Q_scaled, H_scaled, best_degree);
            H_fit = polyval(p, Q_scaled) * H_std + H_mean;
            fit_error = norm(H - H_fit) / norm(H);

            if fit_error > 1e-6
                disp(['Warning: Poor fit for dataset ', num2str(i), ', fit error: ', num2str(fit_error)]);
            end
        catch fit_error
            disp(['Error fitting polynomial for dataset ', num2str(i)]);
            disp(fit_error.message);
            continue;
        end

        pump_data(i).BestDegree = best_degree;
    end

    % Initialize the results array
    D_trimmed = zeros(1, size(QH, 2));

    % Loop through each QH point
    for i = 1:size(QH, 2)
        Q_prime = QH(1, i);
        H_prime = QH(2, i);

        % Find the closest diameter and corresponding curves
        [closest_D_index, closest_H_curve, closest_Q_curve] = find_closest_diameter(Q_prime, H_prime, pump_data);

        if closest_D_index == -1
            error('No valid closest diameter found for Q = %.2f, H = %.2f.', Q_prime, H_prime);
        end

        % Get the best degree for the closest diameter
        best_degree = pump_data(closest_D_index).BestDegree;

        % Get the original untrimmed diameter
        D2 = pump_data(closest_D_index).Diameter;

        % Calculate the new trimmed diameter
        D2_prime = constant_area_scaling(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, best_degree);

        % Store the result
        D_trimmed(i) = D2_prime;
    end
end

function [closest_D_index, closest_H_curve, closest_Q_curve] = find_closest_diameter(Q_prime, H_prime, pump_data)
    min_error = inf;
    closest_D_index = -1;
    closest_H_curve = [];
    closest_Q_curve = [];

    for i = 1:length(pump_data)
        Q_curve = pump_data(i).Q;
        H_curve = pump_data(i).H;

        [~, idx] = min(abs(H_curve - H_prime));
        error = abs(Q_curve(idx) - Q_prime);

        if error < min_error
            min_error = error;
            closest_D_index = i;
            closest_H_curve = H_curve;
            closest_Q_curve = Q_curve;
        end
    end
end

function D2_prime = constant_area_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree)
    % Center and scale the data
    Q_mean = mean(Q_curve);
    Q_std = std(Q_curve);
    H_mean = mean(H_curve);
    H_std = std(H_curve);
    Q_scaled = (Q_curve - Q_mean) / Q_std;
    H_scaled = (H_curve - H_mean) / H_std;
    
    % Fit the polynomial
    p = polyfit(Q_scaled, H_scaled, poly_degree);
    
    % Calculate the scaled A value
    A = (H_prime - H_mean) / H_std / ((Q_prime - Q_mean) / Q_std)^2;

    syms Q
    poly_expr = poly2sym(p, Q);
    
    eqn = A * Q^2 == poly_expr;
    sol = double(solve(eqn, Q));
    
    Q_valid = sol(sol > 0 & imag(sol) == 0 & sol <= max(Q_scaled) & sol >= min(Q_scaled));
    if isempty(Q_valid)
        error('No valid intersection found within the range of the pump curve.');
    end
    Q_intersect = max(Q_valid);
    
    % Scale back the Q_intersect value
    Q_intersect = Q_intersect * Q_std + Q_mean;

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

% 
function compareMethods(QH, D, QH_beps, D_beps, bestTrainedNetH, dataPath)
    % Find all distinct diameters in D
    distinctDiameters = unique(D);

    % Initialize arrays to store MSEs
    mseDiameterNN = zeros(1, length(distinctDiameters));
    mseDiameterCAS = zeros(1, length(distinctDiameters));
    mseQH_bepsNN = zeros(1, length(distinctDiameters));
    mseQH_bepsCAS = zeros(1, length(distinctDiameters));

    % Loop through each distinct diameter
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
        D_temp(indicesToRemove) = [];

        % NN Prediction and MSE calculation
        predictedH_NN = bestTrainedNetH([removedQH(1, :); removedD]);
        mseDiameterNN(dIdx) = mean((removedQH(2, :) - predictedH_NN).^2);

        predictedH_beps_NN = bestTrainedNetH([QH_beps(1, :); D_beps]);
        mseQH_bepsNN(dIdx) = mean((QH_beps(2, :) - predictedH_beps_NN).^2);

        % Constant Area Scaling Prediction and MSE calculation
        D_trimmed = trim_diameters(QH_temp, dataPath);
        predictedH_CAS = interp1(D_trimmed, removedQH(1, :), removedD, 'linear', 'extrap');
        mseDiameterCAS(dIdx) = mean((removedQH(2, :) - predictedH_CAS).^2);

        predictedH_beps_CAS = interp1(D_trimmed, QH_beps(1, :), D_beps, 'linear', 'extrap');
        mseQH_bepsCAS(dIdx) = mean((QH_beps(2, :) - predictedH_beps_CAS).^2);

        % Display results for the current diameter
        fprintf('Diameter: %f\n', diameterToRemove);
        fprintf('MSE (NN, Diameter): %f\n', mseDiameterNN(dIdx));
        fprintf('MSE (NN, QH_beps): %f\n', mseQH_bepsNN(dIdx));
        fprintf('MSE (CAS, Diameter): %f\n', mseDiameterCAS(dIdx));
        fprintf('MSE (CAS, QH_beps): %f\n', mseQH_bepsCAS(dIdx));
        fprintf('--------------------------------------------\n');
    end

    % Save results to a table
    resultTable = table(distinctDiameters, mseDiameterNN', mseDiameterCAS', mseQH_bepsNN', mseQH_bepsCAS', ...
        'VariableNames', {'Diameter', 'MSE_NN_Diameter', 'MSE_CAS_Diameter', 'MSE_NN_QH_beps', 'MSE_CAS_QH_beps'});
    writetable(resultTable, 'comparison_results.csv');
end
