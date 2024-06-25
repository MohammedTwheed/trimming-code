% Merged Script for Centrifugal Pump Impeller Trimming

% Main Script
clear; clc; clf;

dataPath = '../training-data';

% Load data with error handling
try
    [QH, D, QD, P] = loadData(dataPath);
catch ME
    disp(ME.message);
    return;
end

% Extract the Q, H and D is already there in D.
Q = QH(1,:);
H = QH(2,:);

% Get unique diameters
unique_D = unique(D);

% Create a structure to hold Q,H curves for each diameter in unique_D.
pump_data = struct('Diameter', cell(length(unique_D), 1), 'Q', cell(length(unique_D), 1), 'H', cell(length(unique_D), 1));

for i = 1:length(unique_D)
    idx = (D == unique_D(i));
    pump_data(i).Diameter = unique_D(i);
    pump_data(i).Q = Q(idx);
    pump_data(i).H = H(idx);
end

% Curve Fitting using Genetic Algorithm
fitted_models = cell(length(pump_data), 1);
best_degrees = zeros(length(pump_data), 1);

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
        p = polyfit(Q, H, best_degree);
        H_fit = polyval(p, Q);
        fit_error = norm(H - H_fit) / norm(H);
        
        if fit_error > 1e-6
            disp(['Warning: Poor fit for dataset ', num2str(i), ', fit error: ', num2str(fit_error)]);
            fprintf('Q: %s\n', mat2str(Q'));
            fprintf('H: %s\n', mat2str(H'));
            fprintf('Best degree: %d\n', best_degree);
            fprintf('Polynomial coefficients: %s\n', mat2str(p));
            fprintf('Fitted H: %s\n', mat2str(H_fit'));
        end
    catch fit_error
        disp(['Error fitting polynomial for dataset ', num2str(i)]);
        disp(fit_error.message);
        continue;
    end
    
    fitted_models{i} = p;
    best_degrees(i) = best_degree;
end

% Define test points (Q_prime, H_prime)
test_points = [150, 85; 350, 45]; % Example test points
results = zeros(size(test_points, 1), 3);

for i = 1:size(test_points, 1)
    Q_prime = test_points(i, 1);
    H_prime = test_points(i, 2);
    
    % Find the closest diameter based on given point
    [closest_D_index, closest_H_curve, closest_Q_curve] = find_closest_diameter(Q_prime, H_prime, pump_data);
    
    if closest_D_index == -1
        disp('No valid closest diameter found.');
        continue;
    end
    
    % Get the best degree for the closest diameter
    best_degree = best_degrees(closest_D_index);
    
    % Get the original untrimmed diameter
    D2 = pump_data(closest_D_index).Diameter;
    
    % Calculate the new trimmed diameter
    D2_prime = constant_width_scaling(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, best_degree);
    results(i, :) = [Q_prime, H_prime, D2_prime];
    
    % Plot the curves with annotations
    plot_pump_curve(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, D2_prime, best_degree);
end

result_table = array2table(results, 'VariableNames', {'Q_prime', 'H_prime', 'D2_prime'});
disp(result_table);

% Function Definitions
function [QH, D, QD, P] = loadData(dataPath)
    if ~exist(dataPath, 'dir')
        error('Data directory does not exist: %s', dataPath);
    end

    try
        QH = load(fullfile(dataPath, 'QH.mat'));
        D = load(fullfile(dataPath, 'D.mat'));
        QD = load(fullfile(dataPath, 'QD.mat'));
        P = load(fullfile(dataPath, 'Pow.mat'));
    catch ME
        error('Error loading data: %s', ME.message);
    end

    QH = transpose(QH.QH);
    D = transpose(D.D);
    QD = transpose(QD.QD);
    P = transpose(P.P);
end

function fit_error = fitness_polyfit(degree, Q, H)
    try
        p = polyfit(Q, H, degree);
        H_fit = polyval(p, Q);
        fit_error = norm(H - H_fit) / norm(H);
    catch
        fit_error = Inf;
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
    p = polyfit(Q_curve, H_curve, poly_degree);
    A = H_prime / (Q_prime^2);
    
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

function plot_pump_curve(Q_prime, H_prime, H_curve, Q_curve, D2, D2_prime, poly_degree)
    p = polyfit(Q_curve, H_curve, poly_degree);
    Q_fit = linspace(min(Q_curve), max(Q_curve), 100);
    H_fit = polyval(p, Q_fit);
    
    figure;
    plot(Q_curve, H_curve, 'ko', 'MarkerFaceColor', 'k');
    hold on;
    
    plot(Q_fit, H_fit, 'b--', 'LineWidth', 1.5);
    
    A = H_prime / (Q_prime^2);
    fplot(@(Q) A * Q.^2, [min(Q_curve), max(Q_curve)], 'r', 'LineWidth', 1.5);
    
    Q_intersect = Q_prime / D2_prime * D2;
    plot(Q_intersect, A * Q_intersect^2, 'go', 'MarkerFaceColor', 'g');
    
    % Annotations
    plot(Q_prime, H_prime, 'r*', 'MarkerSize', 10); % Test point
    text(Q_prime, H_prime, sprintf('  Test Point (Q=%.1f, H=%.1f)', Q_prime, H_prime), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    text(min(Q_curve), max(H_curve), sprintf('Original Diameter D2 = %.1f', D2), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 12);
    text(Q_intersect, A * Q_intersect^2, sprintf('  New Diameter D2 = %.1f', D2_prime), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    
    legend('Original Data Points', 'Fitted Polynomial Curve', 'Quadratic H = A * Q^2', 'Intersection Point', 'Location', 'best');
    title(sprintf('Intersection of Quadratic Function with Pump Curve for Diameter D2 = %.1f', D2));
    xlabel('Flow Rate (Q)');
    ylabel('Head (H)');
    grid on;
    hold off;
    
    % Plot the new trimmed pump curve
    Q_trimmed = Q_fit * D2_prime / D2;
    H_trimmed = H_fit * (D2_prime / D2)^2;
    
    figure;
    plot(Q_curve, H_curve, 'ko', 'MarkerFaceColor', 'k');
    hold on;
    plot(Q_fit, H_fit, 'b--', 'LineWidth', 1.5);
    plot(Q_trimmed, H_trimmed, 'g-', 'LineWidth', 1.5);
    plot(Q_prime, H_prime, 'r*', 'MarkerSize', 10); % Test point
    text(Q_prime, H_prime, sprintf('  Test Point (Q=%.1f, H=%.1f)', Q_prime, H_prime), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');
    text(min(Q_curve), max(H_curve), sprintf('Original Diameter D2 = %.1f', D2), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 12);
    text(min(Q_trimmed), max(H_trimmed), sprintf('New Diameter D2_prime = %.1f', D2_prime), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', 'FontSize', 12);
    
    legend('Original Data Points', 'Fitted Polynomial Curve', 'Trimmed Pump Curve', 'Location', 'best');
    title(sprintf('Original and Trimmed Pump Curves for Diameter D2 = %.1f and D2_prime  = %.1f', D2, D2_prime));
    xlabel('Flow Rate (Q)');
    ylabel('Head (H)');
    grid on;
    hold off;
end
