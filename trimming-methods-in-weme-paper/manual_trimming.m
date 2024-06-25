% according to "Prediction of the Effect of Impeller Trimming on the 
% Hydraulic Performance of Low Specific-Speed Centrifugal Pumps" 
% paper of D. G. J. Detert Oude Weme et al.
% they discused mainly 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%







% MAIN
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
% Initialize cell arrays to store fitted models
fitted_models = cell(length(pump_data), 1);
best_degrees = zeros(length(pump_data), 1);

% GA options with more aggressive search parameters
ga_options = optimoptions('ga', ...
    'Display', 'off', ...
    'MaxGenerations', 200, ... % Further increase the number of generations
    'MaxStallGenerations', 20, ... % Allow more stall generations
    'FunctionTolerance', 1e-6, ... % Tighten function tolerance even more
    'PopulationSize', 50); % Increase population size significantly

for i = 1:length(pump_data)
    Q = pump_data(i).Q;
    H = pump_data(i).H;
    
    % Check for empty or NaN data
    if isempty(Q) || isempty(H) || any(isnan(Q)) || any(isnan(H))
        disp(['Error: Empty or NaN data for dataset ', num2str(i)]);
        continue; % Skip to the next dataset if data is invalid
    end
    
    % Ensure Q and H are column vectors
    Q = Q(:);
    H = H(:);
    
    % Ensure the length of Q and H are equal
    if length(Q) ~= length(H)
        disp(['Error: Mismatch in length of Q and H for dataset ', num2str(i)]);
        continue;
    end

    % Define fitness function for GA
    fitness_function = @(degree) fitness_polyfit(degree, Q, H);
    
    % Run GA to find the best degree (1 to 5)
    try
        [best_degree, ~] = ga(fitness_function, 1, [], [], [], [], 2, 20, [], ga_options);
    catch ga_error
        disp(['Error optimizing for dataset ', num2str(i)]);
        disp(ga_error.message); % Log GA optimization error message
        continue; % Skip to the next dataset if GA fails
    end
    
    % Ensure best_degree is a positive integer
    best_degree = max(1, round(best_degree));
    
    % Fit the best degree polynomial
    try
        p = polyfit(Q, H, best_degree);
        H_fit = polyval(p, Q);
        fit_error = norm(H - H_fit) / norm(H); % Normalized fitting error
        
        % Ensure the fitting error is sufficiently small
        if fit_error > 1e-6 % Adjust this threshold as needed
            disp(['Warning: Poor fit for dataset ', num2str(i), ', fit error: ', num2str(fit_error)]);
            % Log details of the fit error
            fprintf('Q: %s\n', mat2str(Q'));
            fprintf('H: %s\n', mat2str(H'));
            fprintf('Best degree: %d\n', best_degree);
            fprintf('Polynomial coefficients: %s\n', mat2str(p));
            fprintf('Fitted H: %s\n', mat2str(H_fit'));
        end
    catch fit_error
        disp(['Error fitting polynomial for dataset ', num2str(i)]);
        disp(fit_error.message); % Log fitting error message
        continue; % Skip to the next dataset if fitting fails
    end
    
    % Store the best fit model and degree
    fitted_models{i} = p;
    best_degrees(i) = best_degree;
end


[Q_trad, H_trad, D_trad] = traditionalTrimming(117, 29, pump_data);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Q_trad, H_trad, D_trad] = traditionalTrimming(Q_given, H_given, pump_data)
% this is completly wrong function
    % This function computes Q, H, and D using the traditional method
    % Initialize the output arrays
    Q_trad = zeros(size(Q_given));
    H_trad = zeros(size(H_given));
    D_trad = zeros(size(Q_given));
    
    % Loop through each given Q, H point
    for i = 1:length(Q_given)
        Qg = Q_given(i);
        Hg = H_given(i);
        
        % Find the nearest pump curve based on the given Q and H
        min_error = inf;
        best_curve = struct('Q', [], 'H', [], 'D', []);
        
        for j = 1:length(pump_data)
            Q_curve = pump_data(j).Q;
            H_curve = pump_data(j).H;
            D_curve = pump_data(j).Diameter;
            
            % Calculate the error based on the given Q, H
            [~, idx] = min(abs(H_curve - Hg));
            error = abs(Q_curve(idx) - Qg);
            
            if error < min_error
                min_error = error;
                best_curve.Q = Q_curve;
                best_curve.H = H_curve;
                best_curve.D = D_curve;
            end
        end
        
        % Use the best curve to find the trimmed Q, H, and D
        D1 = best_curve.D;
        Q1 = Qg;
        H1 = Hg;
        
        % Fit a polynomial to the best curve
        p = polyfit(best_curve.Q, best_curve.H, 2); % Quadratic fit
        
        % Solve H = (H_given / Q_given^2) Q^2 using the polynomial
        syms Q_sym;
        H_eq = p(1) * Q_sym^2 + p(2) * Q_sym + p(3);
        Q_sol = double(solve(H_eq == (H1 / Q1^2) * Q_sym^2, Q_sym));
        
        % Select the positive real solution
        Q2 = Q_sol(Q_sol > 0);
        if isempty(Q2)
            Q2 = NaN; % In case no valid solution is found
        else
            Q2 = Q2(1); % Select the first positive solution
        end
        
        % Calculate the corresponding H2 using the polynomial
        H2 = polyval(p, Q2);
        
        % Calculate the trimmed diameter D2
        D2 = D1 * (Q2 / Q1);
        
        % Store the results
        Q_trad(i) = Q2;
        H_trad(i) = H2;
        D_trad(i) = D2;
    end
end


% Define fitness function for GA
function fit_error = fitness_polyfit(degree, Q, H)
    try
        p = polyfit(Q, H, degree);
        H_fit = polyval(p, Q);
        fit_error = norm(H - H_fit) / norm(H); % Normalized fitting error
    catch
        fit_error = Inf; % Return a large value if fitting fails
    end
end

function [QH, D, QD, P] = loadData(dataPath)
% loadData: Loads data from files into MATLAB variables. Inputs:
%   dataPath - Path to the directory containing the data files.
% Outputs:
%   QH - Q flowrate and Head data (corresponding to D diameters).
%   (matrix) D - Diameters. (matrix) QD - Q flowrate and diameters
%   (corresponding to power in P). (matrix) P - Power values. (matrix)

% Validate data path
if ~exist(dataPath, 'dir')
    error('Data directory does not exist: %s', dataPath);
end

% Load data from files with error handling
try
    QH = load(fullfile(dataPath, 'QH.mat'));
    D = load(fullfile(dataPath, 'D.mat'));
    QD = load(fullfile(dataPath, 'QD.mat'));
    P = load(fullfile(dataPath, 'Pow.mat'));
catch ME
    error('Error loading data: %s', ME.message);
end

% Extract desired variables directly (note that this according to our own
% data only)
QH = transpose(QH.QH); %  QH struct contains a single variable named 'QH'
D = transpose(D.D);    %  D struct contains a single variable named 'D'
QD = transpose(QD.QD); %  QD struct contains a single variable named 'QD'
P = transpose(P.P);    %  P struct contains a single variable named 'P'

% the transpose here b.c the of the way train() function in matlab
% interpret the input output featrues see documentation for more info
% without transpose it would treat the whole vector or matrix as one
% input feature.
end
