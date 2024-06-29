
% Initialize minimum AvgMSE to a large value
minAvgMSE = inf;
minIndex = -1;

% Loop through each cell in the array
for i = 1:length(local_nn_logs)
    % Access the struct in the current cell
    currentStruct = local_nn_logs{i};
    
    % Extract the AvgMSE field from the struct
    currentAvgMSE = currentStruct.AvgMSE;
    
    % Check if the current AvgMSE is less than the minimum AvgMSE found so far
    if currentAvgMSE < minAvgMSE
        % Update the minimum AvgMSE and the index of the struct
        minAvgMSE = currentAvgMSE;
        minIndex = i;
    end
end

% Display the minimum AvgMSE and the index of the struct
fprintf('Minimum AvgMSE: %f found at index: %d\n', minAvgMSE, minIndex);

% Extract all elements in the struct at the found index
bestStruct = local_nn_logs{minIndex};

% Display or use the fields of the best struct
disp('Fields of the struct with the minimum AvgMSE:');
disp(bestStruct);