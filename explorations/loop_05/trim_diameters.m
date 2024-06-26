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
        D2_prime = constant_width_scaling(Q_prime, H_prime, closest_H_curve, closest_Q_curve, D2, best_degree);

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

function D2_prime = constant_width_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree)
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
    A = H_prime/ (Q_prime)^2;

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

