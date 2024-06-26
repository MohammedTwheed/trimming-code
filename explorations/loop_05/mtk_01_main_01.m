%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; clf;

% Load data
load('filtered_QHD_table.mat');
load('filtered_QDP_table.mat');
load('deleted_QHD_table.mat');
load('deleted_QDP_table.mat');

QH = [filtered_QHD_table.FlowRate_m3h, filtered_QHD_table.Head_m]';
D  = [filtered_QHD_table.Diameter_mm]';

QH_beps = [deleted_QHD_table.FlowRate_m3h, deleted_QHD_table.Head_m]';
D_beps  = [deleted_QHD_table.Diameter_mm]';

QD = [filtered_QDP_table.FlowRate_m3h, filtered_QDP_table.Diameter_mm]';
P  = [filtered_QDP_table.Power_kW]';

QD_beps = [deleted_QDP_table.FlowRate_m3h, deleted_QDP_table.Diameter_mm]';
P_beps  = [deleted_QDP_table.Power_kW]';

% Hyperparameters based on latest optimization with GA
randomSeed = 4837;
nn_size_matrix = [2, 15];
maxEpochs = 191;
trainFcn = 'trainlm';

% Train neural networks
[trainedNetQHD, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QH, D, randomSeed);
[trainedNetQDH, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, [QH(1, :); D], QH(2, :), randomSeed);
[trainedNetQDP, ~, ~, ~, ~] = train_nn(nn_size_matrix, maxEpochs, trainFcn, QD, P, randomSeed);

% Process data and visualize
processDataAndVisualize(QH, D, QD, P, trainedNetQDH, trainedNetQDP, true);

% Trim diameters using neural network and constant width scaling
D_trimmed_nn = trim_diameters(QH, 'filtered_QHD_table.mat');
D_trimmed_cws = constant_width_scaling_method(QH, D);

% Compare errors between NN and traditional trimming methods
compare_trimming_methods(QH, D, D_trimmed_nn, D_trimmed_cws);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

function D_trimmed_nn = trim_diameters(QH, dataPath)
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
    D_trimmed_nn = zeros(1, size(QH, 2));
    for i = 1:size(QH, 2)
        Q_prime = QH(1, i);
        H_prime = QH(2, i);
        [closest_D_index, closest_H_curve, closest_Q_curve] = find_closest_diameter(Q_prime, H_prime, pump_data);
        if closest_D_index == -1
            error('No valid closest diameter found for Q = %.2f, H = %.2f.', Q_prime, H_prime);
        end
        best_degree = pump_data(closest_D_index).BestDegree;
        D2 = pump_data(closest_D_index).Diameter;
        D2_prime = constant_width_scaling(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, best_degree);
        D_trimmed_nn(i) = D2_prime;
    end
end

function D_trimmed_cws = constant_width_scaling_method(QH, D)
    % Implement the Constant Width Scaling method
    D_trimmed_cws = zeros(1, size(QH, 2));
    for i = 1:size(QH, 2)
        Q = QH(1, i);
        H = QH(2, i);
        D_trimmed_cws(i) = (Q / H) * D(i); % Example scaling logic, replace with actual method
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

function D2_prime = constant_width_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree)
    Q_mean = mean(Q_curve);
    Q_std = std(Q_curve);
    H_mean = mean(H_curve);
    H_std = std(H_curve);
    Q_scaled = (Q_curve - Q_mean) / Q_std;
    H_scaled = (H_curve - H_mean) / H_std;
    p = polyfit(Q_scaled, H_scaled, poly_degree);
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
    Q_intersect = Q_intersect * Q_std + Q_mean;
    D2_prime = Q_prime / Q_intersect * D2;
end

function error = fitness_polyfit(degree, Q, H)
    degree = round(degree); 
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

function processDataAndVisualize(QH, D, QD, P, bestTrainedNetD, bestTrainedNetP, saveFigures)
    if nargin < 7
        saveFigures = true;
    end
    QH = QH';
    QD = QD';
    
    [Qq, Hq] = meshgrid(0:2:440, 0:.5:90);
    Dq = griddata(QH(:, 1), QH(:, 2), sim(bestTrainedNetD, QH')', Qq, Hq);
    
    [QDq, Pq] = meshgrid(0:2:440, 220:.5:270);
    DqP = griddata(QD(:, 1), QD(:, 2), sim(bestTrainedNetP, QD')', QDq, Pq);
    
    figure;
    subplot(2, 1, 1);
    mesh(Qq, Hq, Dq);
    xlabel('Flow Rate (m^3/h)');
    ylabel('Head (m)');
    zlabel('Diameter (mm)');
    title('Neural Network vs Data Points (Diameters)');
    hold on;
    scatter3(QH(:, 1), QH(:, 2), D, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    legend('Neural Network', 'Data Points');
    hold off;
    
    subplot(2, 1, 2);
    mesh(QDq, Pq, DqP);
    xlabel('Flow Rate (m^3/h)');
    ylabel('Diameter (mm)');
    zlabel('Power (kW)');
    title('Neural Network vs Data Points (Power)');
    hold on;
    scatter3(QD(:, 1), QD(:, 2), P, 'o', 'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');
    legend('Neural Network', 'Data Points');
    hold off;
    
    if saveFigures
        currentTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
        saveas(gcf, ['diameter_power_visualization_' char(currentTime) '.fig']);
        saveas(gcf, ['diameter_power_visualization_' char(currentTime) '.png']);
    end
end

function compare_trimming_methods(QH, D, D_trimmed_nn, D_trimmed_cws)
    % Calculate errors
    mse_nn = mean((D - D_trimmed_nn).^2);
    mse_cws = mean((D - D_trimmed_cws).^2);

    % Display errors
    fprintf('Mean Squared Error (NN): %.4f\n', mse_nn);
    fprintf('Mean Squared Error (CWS): %.4f\n', mse_cws);

    % Plot comparison
    figure;
    hold on;
    plot(D, 'k', 'DisplayName', 'Actual Diameter');
    plot(D_trimmed_nn, 'b--', 'DisplayName', 'NN Trimmed Diameter');
    plot(D_trimmed_cws, 'r-.', 'DisplayName', 'CWS Trimmed Diameter');
    legend('show');
    xlabel('Sample Index');
    ylabel('Diameter (mm)');
    title('Comparison of Trimmed Diameters');
    hold off;

    % Save comparison plot
    currentTime = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
    saveas(gcf, ['trimmed_diameter_comparison_' char(currentTime) '.fig']);
    saveas(gcf, ['trimmed_diameter_comparison_' char(currentTime) '.png']);
end
