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

% Calculate the best polynomial degree for each diameter
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
        disp(['Error: Length mismatch between Q and H for dataset ', num2str(i)]);
        continue;
    end
    best_degree = ga(@(deg) fit_polynomial(Q, H, deg), 1, [], [], [], [], 1, 10, [], ga_options);
    pump_data(i).BestDegree = best_degree;
end

% Train and test NNs by removing one diameter at a time
for dIdx = 1:length(distinctDiameters)
    % Current diameter to remove
    diameterToRemove = distinctDiameters(dIdx);

    % Find indices of the current diameter in D
    indicesToRemove = find(D == diameterToRemove);

    % Remove rows from QH and D based on the indices
    QH_temp = QH;
    D_temp = D;
    QH_temp(:, indicesToRemove) = [];
    D_temp(:, indicesToRemove) = [];

    % Remove rows from QD and P based on the indices
    indicesToRemoveQD = find(QD(2,:) == diameterToRemove);
    QD_temp = QD;
    P_temp = P;
    QD_temp(:, indicesToRemoveQD) = [];
    P_temp(indicesToRemoveQD) = [];

    % Train neural networks
    [trainedNetQHD, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QH_temp, D_temp, randomSeed);
    [trainedNetQDH, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, [QH_temp(1,:); D_temp], QH_temp(2,:), randomSeed);
    [trainedNetQDP, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QD_temp, P_temp, randomSeed);

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

function fit_error = fit_polynomial(Q, H, degree)
    best_degree = max(2, round(degree)); % Ensure the degree is at least 2 and an integer.
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

end

function errorResults = compareManualMethods(QH_beps, D_beps, trainedNetQHD, pump_data, diameterToRemove)
    % Predict diameters for BEP points using the trained neural network
    predicted_D_beps = trainedNetQHD(QH_beps);
    
    % Initialize error storage
    errors = [];
    
    % Iterate through each unique diameter
    for i = 1:length(pump_data)
        if pump_data(i).Diameter == diameterToRemove
            continue; % Skip the removed diameter
        end
        
        % Extract Q-H data for the current diameter
        Q = pump_data(i).Q;
        H = pump_data(i).H;
        
        % Predict diameter using manual method (example: polynomial fitting)
        manual_diameter = predict_diameter_manual(Q, H, pump_data(i).BestDegree);
        
        % Calculate error between manual method and neural network predictions
        manual_error = mean(abs(manual_diameter - D_beps(i)));
        nn_error = mean(abs(predicted_D_beps - D_beps(i)));
        
        % Store the errors
        errors = [errors; manual_error, nn_error];
    end
    
    errorResults = mean(errors, 1);
end

function diameter = predict_diameter_manual(Q, H, degree)
    % Fit a polynomial to Q-H data
    degree = max(1, round(degree)); % Ensure the degree is at least 1 and an integer.
    p = polyfit(Q, H, degree);
    
    % Predict diameter using the polynomial coefficients
    diameter = polyval(p, Q);
end

function displayResults(errorResults)
    disp('Error Results:');
    disp(errorResults);
end

function saveResults(errorResults)
    save('errorResults.mat', 'errorResults');
end

function visualizeResults(QH, D, QD, P, trainedNetQHD, trainedNetQDP)
    % Create a figure for the results
    figure;

    % Plot Q-H-D curves
    subplot(2, 2, 1);
    scatter3(QH(1,:), QH(2,:), D, 'b');
    hold on;
    predicted_D = trainedNetQHD(QH);
    scatter3(QH(1,:), QH(2,:), predicted_D, 'r');
    xlabel('Flow Rate (m^3/h)');
    ylabel('Head (m)');
    zlabel('Diameter (mm)');
    legend('Original Data', 'NN Predictions');
    title('Q-H-D Curves');

    % Plot Q-D-P curves
    subplot(2, 2, 2);
    scatter3(QD(1,:), QD(2,:), P, 'b');
    hold on;
    predicted_P = trainedNetQDP(QD);
    scatter3(QD(1,:), QD(2,:), predicted_P, 'r');
    xlabel('Flow Rate (m^3/h)');
    ylabel('Diameter (mm)');
    zlabel('Power (kW)');
    legend('Original Data', 'NN Predictions');
    title('Q-D-P Curves');
end
