
% Example untrimmed pump curve data from dr farid exam
% 2006 from old exams in from-turbomachinery-course
% folder in topics.
Q_curve = [50, 100, 150, 200, 250, 300]; % Flow rates
H_curve = [200, 192, 180, 160, 127, 72]; % Corresponding heads
% Original untrimmed diameter
D2 = 550;
% Define test points (Q_prime, H_prime)
test_points = [150, 125];
% Polynomial fit degree
poly_degree = 4;  
% Preallocate results
num_points = size(test_points, 1);
results = zeros(num_points, 3);
% Calculate D2' for each test point
for i = 1:num_points
    Q_prime = test_points(i, 1);
    H_prime = test_points(i, 2);
    D2_prime = constant_width_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree);
    results(i, :) = [Q_prime, H_prime, D2_prime];
end
% Create a table
result_table = array2table(results, 'VariableNames', {'Q_prime', 'H_prime', 'D2_prime'});
% Display the table
disp(result_table);

function D2_prime = constant_width_scaling(Q_prime, H_prime, H_curve, Q_curve, D2, poly_degree)
    % Fit a polynomial to the original pump curve
    p = polyfit(Q_curve, H_curve, poly_degree);
    
    % Calculate the coefficient A from the given H' and Q'
    A = H_prime / (Q_prime^2);
    
    % Define the polynomial coefficients symbolically
    syms Q
    poly_expr = poly2sym(p, Q);
    
    % Solve for the intersection using symbolic solver
    eqn = A * Q^2 == poly_expr;
    sol = double(solve(eqn, Q));
    
    % Select the realistic root (positive and within the range)
    Q_valid = sol(sol > 0 & imag(sol) == 0 & sol <= max(Q_curve) & sol >= min(Q_curve));
    if isempty(Q_valid)
        error('No valid intersection found within the range of the pump curve.');
    end
    Q_intersect = max(Q_valid);
    
    % Calculate the trimmed diameter D2'
    D2_prime = Q_prime / Q_intersect * D2;
    
    % Plotting for visualization
    figure;
    
    % Plot original pump data points
    plot(Q_curve, H_curve, 'ko', 'MarkerFaceColor', 'k');
    hold on;
    
    % Plot fitted polynomial curve
    Q_fit = linspace(min(Q_curve), max(Q_curve), 100);
    H_fit = polyval(p, Q_fit);
    plot(Q_fit, H_fit, 'b--', 'LineWidth', 1.5);
    
    % Plot quadratic function
    fplot(@(Q) A * Q.^2, [min(Q_curve), max(Q_curve)], 'r', 'LineWidth', 1.5);
    
    % Plot intersection point
    plot(Q_intersect, A * Q_intersect^2, 'go', 'MarkerFaceColor', 'g');
    
    % Annotations and legends
    legend('Original Data Points', 'Fitted Polynomial Curve', 'Quadratic H = A * Q^2', 'Intersection Point', 'Location', 'best');
    title('Intersection of Quadratic Function with Pump Curve');
    xlabel('Flow Rate (Q)');
    ylabel('Head (H)');
    grid on;
    hold off;
end

