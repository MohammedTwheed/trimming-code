clear; clc; clf;

% Load data
load('filtered_QHD_table.mat');
load('filtered_QDP_table.mat');
load('deleted_QHD_table.mat');
load('deleted_QDP_table.mat');

% Extract data
QH = [filtered_QHD_table.FlowRate_m3h, filtered_QHD_table.Head_m]';
D  = [filtered_QHD_table.Diameter_mm]';

QH_beps = [deleted_QHD_table.FlowRate_m3h, deleted_QHD_table.Head_m]';
D_beps  = [deleted_QHD_table.Diameter_mm]';

QD = [filtered_QDP_table.FlowRate_m3h, filtered_QDP_table.Diameter_mm]';
P  = [filtered_QDP_table.Power_kW]';

QD_beps = [deleted_QDP_table.FlowRate_m3h, deleted_QDP_table.Diameter_mm]';
P_beps  = [deleted_QDP_table.Power_kW]';

% Hyperparameters
randomSeed = 4837;
nn_size_matrix = [2, 16];
maxEpochs = 191;
trainFcn = 'trainlm';

% Initialize result storage
errorResults = [];

% Find all distinct diameters in D

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


% Train and test NNs by removing one diameter at a time
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

    % Train neural networks
    [trainedNetQHD, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QH_temp, D_temp, randomSeed);
    [trainedNetQDH, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, [QH_temp(1,:); D_temp], QH_temp(2,:), randomSeed);
    [trainedNetQDP, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QD, P, randomSeed);

    % Compare NN with manual methods
    errors = compareManualMethods(QH_beps, D_beps, trainedNetQHD, pump_data, diameterToRemove);

    % Append results
    errorResults = [errorResults; errors];
end

% Display and save results
displayResults(errorResults);
saveResults(errorResults);

% Visualize results
visualizeResults(QH', D', QD', P', trainedNetQHD, trainedNetQDP);

% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [trainedNet, avgMSEs, trainPerformance, valPerformance, testPerformance] = train_nn(nn_size_matrix, maxEpochs, trainFcn, x, t, randomSeed)
    rng(randomSeed); % Set random seed for reproducibility.
    net = feedforwardnet(nn_size_matrix, trainFcn);
    net.trainParam.showWindow = false;
    net.trainParam.epochs = maxEpochs;
    net.performFcn = 'mse';
    net.input.processFcns = {'removeconstantrows', 'mapminmax'};
    net.output.processFcns = {'removeconstantrows', 'mapminmax'};
    net.divideFcn = 'dividerand';
    net.divideMode = 'sample';
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

function D_trimmed = trim_diameters(QH, dataPath)
    % Load the data
    loadedData = load(dataPath);
    filtered_QHD_table = loadedData.filtered_QHD_table;
    Q = filtered_QHD_table.FlowRate_m3h;
    H = filtered_QHD_table.Head_m;
    D = filtered_QHD_table.Diameter_mm;
    unique_D = unique(D);
    pump_data = struct('Diameter', cell(length(unique_D), 1), 'Q', cell(length(unique_D), 1), 'H', cell(length(unique_D), 1), 'BestDegree', cell(length(unique_D), 1));
    for i = 1:length(unique_D)
        idx = (D == unique_D(i));
        pump_data(i).Diameter = unique_D(i);
        pump_data(i).Q = Q(idx);
        pump_data(i).H = H(idx);
    end
    ga_options = optimoptions('ga', 'Display', 'off', 'MaxGenerations', 200, 'MaxStallGenerations', 20, 'FunctionTolerance', 1e-6, 'PopulationSize', 50);
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
    D_trimmed = zeros(1, size(QH, 2));
    for i = 1:size(QH, 2)
        Q_prime = QH(1, i);
        H_prime = QH(2, i);
        [closest_D_index, closest_H_curve, closest_Q_curve] = find_closest_diameter(Q_prime, H_prime, pump_data);
        if closest_D_index == -1
            error('No valid closest diameter found for Q = %.2f, H = %.2f.', Q_prime, H_prime);
        end
        best_degree = pump_data(closest_D_index).BestDegree;
        D2 = pump_data(closest_D_index).Diameter;
        D2_prime = constant_area_scaling(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, best_degree);
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

function D2_prime = constant_area_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, degree)
    if degree < 1
        error('Invalid polynomial degree: %d. Degree must be at least 1.', degree);
    end
    if degree == 1
        error('Constant area scaling is not applicable for linear curves.');
    end
    p = polyfit(Q_curve, H_curve, degree);
    H_max = polyval(p, Q_prime);
    D2_prime = D2 * sqrt(H_prime / H_max);
end

function [errors] = compareManualMethods(QH_beps, D_beps, trainedNetQHD, pump_data, removedDiameter)
    % Initialize error array
    errors = [];

    % Predict with neural network
    predictedHead_NN = trainedNetQHD(QH_beps);

    % Calculate errors
    error_NN = sqrt(mean((predictedHead_NN - D_beps).^2));

    % Compare with manual methods and append to errors
    errors = [errors; removedDiameter, error_NN];
end

function displayResults(errorResults)
    % Display results in a formatted table
    disp('Diameter Removed | NN Error');
    disp('-----------------|----------');
    for i = 1:size(errorResults, 1)
        fprintf('%17d | %8.4f\n', errorResults(i, 1), errorResults(i, 2));
    end
end

function saveResults(errorResults)
    % Save results to a CSV file
    csvwrite('mtk_02_main_01_error_results.csv', errorResults);
end

function visualizeResults(QH, D, QD, P, trainedNetQHD, trainedNetQDP)
    % Create a figure for the results
    figure;

    % Plot Q-H-D curves
    subplot(2, 2, 1);
    scatter3(QH(:,1), QH(:,2), D, 'b');
    hold on;
    predicted_D = trainedNetQHD(QH');
    scatter3(QH(:,1), QH(:,2), predicted_D, 'r');
    xlabel('Flow Rate (m^3/h)');
    ylabel('Head (m)');
    zlabel('Diameter (mm)');
    legend('Original Data', 'NN Predictions');
    title('Q-H-D Curves');

    % Plot Q-D-P curves
    subplot(2, 2, 2);
    scatter3(QD(:,1), QD(:,2), P, 'b');
    hold on;
    predicted_P = trainedNetQDP(QD');
    scatter3(QD(:,1), QD(:,2), predicted_P, 'r');
    xlabel('Flow Rate (m^3/h)');
    ylabel('Diameter (mm)');
    zlabel('Power (kW)');
    legend('Original Data', 'NN Predictions');
    title('Q-D-P Curves');
end


