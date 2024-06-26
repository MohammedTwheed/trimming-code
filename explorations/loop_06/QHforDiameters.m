%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN SCRIPT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; clc; clf;

% Load data
dataPath = 'G:\AlexUniv\alex-univ-4.2\projects\mtk-bachelor-project\topics\trimming-code\explorations\loop_06';  
[QH, D, QD, P, QH_beps, D_beps] = loadData(dataPath);

% Define parameters
userSeed = 4826;
mseThreshold = 0.000199;
weightDiameter = 0.5;
weightBeps = 0.5;
distinctDiameters = unique(D);

% Initialize result matrix
results = [];

% Loop over distinct diameters
for dIdx = 1:length(distinctDiameters)
    diameterToRemove = distinctDiameters(dIdx);
    indicesToRemove = find(D == diameterToRemove);
    
    % Prepare data
    QH_temp = QH;
    D_temp = D;
    QH_temp(:, indicesToRemove) = [];
    D_temp(:, indicesToRemove) = [];
    removedQH = QH(:, indicesToRemove);
    removedD = D(indicesToRemove);

    % Optimize neural network hyperparameters
    lowerBounds = [2, 13, 13, 1, 1];
    upperBounds = [2, 300, 300, 2, 1];
    prevCombinedMSE = inf;

    for i = 1:20
        [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, combinedMSE, mseDiameter, mseQH_beps, nnPerfVect] = ...
            optimizeNNForTrimmingPumpImpeller([QH_temp(1, :); D_temp], QH_temp(2, :), userSeed + i, lowerBounds, upperBounds, removedQH, removedD, QH_beps, D_beps, weightDiameter, weightBeps);
        
        deltaMSE = prevCombinedMSE - combinedMSE;

        % Adjust bounds
        [lowerBounds, upperBounds] = adjustBounds(deltaMSE, lowerBounds, upperBounds, optimalHyperParams);

        % Update previous combined MSE
        prevCombinedMSE = combinedMSE;

        % Plot and save results
        plotAndSaveResults(removedQH, distinctDiameters, diameterToRemove, i, optimalHyperParams, bestTrainedNet, mseDiameter, nnPerfVect);
        
        % Save results
        results = [results; i, optimalHyperParams, finalMSE, randomSeed, nnPerfVect, combinedMSE, mseDiameter, mseQH_beps];
        
        % Early exit
        if (mseDiameter < mseThreshold) && (nnPerfVect(3) < 0.0199) && (mseQH_beps < mseThreshold)
            fprintf('MSE for diameter %d is below the threshold. Exiting loop.\n', diameterToRemove);
            break;
        end
    end
end

% Write results to CSV
header = ["Iteration", "Hidden Layer 1 Size", "Hidden Layer 2 Size", "Max Epochs", "Training Function", "Activation Function", "Final MSE", "Random Seed", "Training Error", "Validation Error", "Test Error", "Combined MSE", "MSE Diameter", "MSE QH beps"];
writematrix([header; results], './01/results_loop.csv');

disp('./01/Results saved to results_loop.csv');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [QH, D, QD, P, QH_beps, D_beps] = loadData(dataPath)
    % loadData: Loads data from files into MATLAB variables.
    if ~exist(dataPath, 'dir')
        error('Data directory does not exist: %s', dataPath);
    end
    
    try
        load(fullfile(dataPath, 'filtered_QHD_table.mat'));
        load(fullfile(dataPath, 'filtered_QDP_table.mat'));
       load(fullfile(dataPath, 'deleted_QHD_table.mat'));
         load(fullfile(dataPath, 'deleted_QDP_table.mat'));
    catch ME
        error('Error loading data: %s', ME.message);
    end
    
    QH = [filtered_QHD_table.FlowRate_m3h, filtered_QHD_table.Head_m]';
    D = [filtered_QHD_table.Diameter_mm]';
    QD = [filtered_QDP_table.FlowRate_m3h, filtered_QDP_table.Diameter_mm]';
    P = [filtered_QDP_table.Power_kW]';
    
    % Add QH_beps and D_beps from the deleted tables
    QH_beps = [deleted_QHD_table.FlowRate_m3h, deleted_QHD_table.Head_m]';
    D_beps = [deleted_QHD_table.Diameter_mm]';
end

function [optimalHyperParams, finalMSE, randomSeed, bestTrainedNet, combinedMSE, mseDiameter, mseQH_beps, nnPerfVect] = optimizeNNForTrimmingPumpImpeller(x, t, userSeed, lowerBounds, upperBounds, removedQH, removedD, QH_beps, D_beps, weightDiameter, weightBeps)
    if nargin < 3
        randomSeed = randi(10000);
    else
        randomSeed = userSeed;
    end

    tic;
    disp("Optimization exploration_02 in progress. This process may take up to 30 seconds...");

    trainingFunctionOptions = {'trainlm', 'trainbr', 'trainrp', 'traincgb', 'traincgf', 'traincgp', 'traingdx', 'trainoss'};
    activationFunctionOptions = {'tansig', 'logsig'};

    gaOptions = optimoptions('ga', ...
        'PopulationSize', 17, ...
        'MaxGenerations', 13, ...
        'CrossoverFraction', 0.8, ...
        'MutationFcn', @dynamicMutation, ...
        'CrossoverFcn', @dynamicCrossover, ...
        'ConstraintTolerance', 0.000991, ...
        'FitnessLimit', 0.000991, ...
        'EliteCount', 2, ...
        'Display', 'iter', ...
        'UseParallel', true);

    global bestTrainedNet;
    bestTrainedNet = [];
    global nnPerfVect;
    nnPerfVect = [];

    function [avgMSEs] = evaluateHyperparameters(hyperParams, x, t, randomSeed)
        rng(randomSeed);
        hiddenLayer1Size = round(hyperParams(1));
        hiddenLayer2Size = round(hyperParams(2));
        maxEpochs = round(hyperParams(3));
        trainingFunctionIdx = round(hyperParams(4));
        activationFunctionIdx = round(hyperParams(5));

        net = feedforwardnet([hiddenLayer1Size, hiddenLayer2Size], trainingFunctionOptions{trainingFunctionIdx});
        net.trainParam.showWindow = false;
        net.trainParam.epochs = maxEpochs;
        net.layers{1}.transferFcn = activationFunctionOptions{activationFunctionIdx};
        net.layers{2}.transferFcn = activationFunctionOptions{activationFunctionIdx};
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

        mseDiameter = evaluateMSE(trainedNet, removedQH, removedD);
        mseQH_beps = evaluateMSE(trainedNet, QH_beps, D_beps);

        combinedMSE = weightDiameter * mseDiameter + weightBeps * mseQH_beps;

        if isempty(bestTrainedNet) || avgMSEs < perform(bestTrainedNet, t, bestTrainedNet(x))
            bestTrainedNet = trainedNet;
            nnPerfVect = [trainPerformance, valPerformance, testPerformance];
        end

        avgMSEs = combinedMSE;
    end

    rng(randomSeed);
    [optimalHyperParams, finalMSE] = ga(@(hyperParams) evaluateHyperparameters(hyperParams, x, t, randomSeed), ...
        length(lowerBounds), [], [], [], [], lowerBounds, upperBounds, [], gaOptions);

    optimalHyperParams = round(optimalHyperParams);
    elapsedTime = toc;

    trainingFunction = trainingFunctionOptions{optimalHyperParams(4)};
    activationFunction = activationFunctionOptions{optimalHyperParams(5)};

    logFile = 'optimizeNNForTrimmingPumpImpeller_log.txt';
    fid = fopen(logFile, 'a');
    if fid == -1
        error('Error opening log file for appending.');
    end

    currentTime = datetime('now', 'Format', 'yyyy-MM-dd HH:MM:SS');
    fprintf(fid, '%s\n', char(currentTime));
    fprintf(fid, 'Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d, Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
        optimalHyperParams(1), optimalHyperParams(2), optimalHyperParams(3), trainingFunction, activationFunction);
    fprintf(fid, 'Final Mean Squared Error (MSE): %.4f\n', finalMSE);
    fprintf(fid, 'Random Seed Used: %d\n', randomSeed);
    fprintf(fid, 'Optimization Duration: %.4f seconds\n', elapsedTime);
    fprintf(fid, 'Combined MSE: %.4f, MSE Diameter: %.4f, MSE QH beps: %.4f\n', combinedMSE, mseDiameter, mseQH_beps);
    fprintf(fid, 'Training Error: %.4f, Validation Error: %.4f, Test Error: %.4f\n\n', nnPerfVect(1), nnPerfVect(2), nnPerfVect(3));
    fclose(fid);

    fprintf('Optimized Hyperparameters: Hidden Layer 1 Size = %d, Hidden Layer 2 Size = %d, Max Epochs = %d, Training Function = %s, Activation Function = %s\n', ...
        optimalHyperParams(1), optimalHyperParams(2), optimalHyperParams(3), trainingFunction, activationFunction);
    fprintf('Final Mean Squared Error (MSE): %.4f\n', finalMSE);
    fprintf('Random Seed Used: %d\n', randomSeed);
    fprintf('Optimization Duration: %.4f seconds\n', elapsedTime);

    combinedMSE = finalMSE;
end

function mse = evaluateMSE(net, QH, D)
    predictions = net([QH(1, :); D])';
    mse = mean((QH(2, :)' - predictions).^2 / sum(QH(2, :)));
end

function [lowerBounds, upperBounds] = adjustBounds(deltaMSE, lowerBounds, upperBounds, optimalHyperParams)
    if deltaMSE > 0.01
        adjustment = [0, 5, 15, 0, 0];
    elseif deltaMSE > 0.001
        adjustment = [0, 2, 10, 0, 0];
    else
        adjustment = [0, 1, 5, 0, 0];
    end
    lowerBounds = max(lowerBounds, [2, optimalHyperParams(2), optimalHyperParams(3), 1, 1] - adjustment);
    upperBounds = min(upperBounds, [2, optimalHyperParams(2), optimalHyperParams(3), 2, 1] + adjustment);
end

function plotAndSaveResults(removedQH, distinctDiameters, diameterToRemove, iteration, optimalHyperParams, bestTrainedNet, mseDiameter, nnPerfVect)
    figure;
    hold on;
    scatter(removedQH(1, :)', removedQH(2, :)', 'r', 'filled', 'DisplayName', sprintf('Removed Diameter: %dmm', diameterToRemove));
    
    Qt = linspace(0, 400, 200);
    
    for diameterIndex = 1:length(distinctDiameters)
        desiredDiameter = distinctDiameters(diameterIndex);
        Dt = repmat(desiredDiameter, length(Qt), 1);
        filteredQH = bestTrainedNet([Qt; Dt'])';
        
        legendLabel = strcat('Diameter: ', num2str(desiredDiameter), 'mm');
        plot(Qt, filteredQH, 'DisplayName', legendLabel);
        text(Qt(end), filteredQH(end), sprintf('%dmm', desiredDiameter), 'FontSize', 8, 'Color', 'black', 'BackgroundColor', 'white');
    end
    
    scatter(Qa', Ha', 'b', 'filled', 'DisplayName', 'Reference Points');
    
    xlabel('Q (m^3/h)');
    ylabel('H (m)');
    title(['(Q, H) slices with Diameters, Removed Diameter: ' num2str(diameterToRemove) 'mm']);
    legend;
    hold off;
    
    filename = sprintf('../loop_05/01/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%d_test-%d.png', diameterToRemove, iteration, ...
        optimalHyperParams(1), optimalHyperParams(2), optimalHyperParams(3), ...
        optimalHyperParams(4), optimalHyperParams(5), mseDiameter, nnPerfVect(3));
    saveas(gcf, filename);
    
    filename = sprintf('../loop_05/01/nn_diameter-%d_iteration_%d_%d-%d-%d-%d-%d_mseDia-%d_test-%d.mat', diameterToRemove, iteration, ...
        optimalHyperParams(1), optimalHyperParams(2), optimalHyperParams(3), ...
        optimalHyperParams(4), optimalHyperParams(5), mseDiameter, nnPerfVect(3));
    save(filename, 'bestTrainedNet');
    close(gcf);
end

function processDataAndVisualize(QH, D, QD, P, bestTrainedNetD, bestTrainedNetP, saveFigures)
    if nargin < 7
        saveFigures = true;
    end
    
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

function mutationChildren = dynamicMutation(parents, options, nvars, FitnessFcn, state, thisScore, thisPopulation)
    % Check if state has the Generation field
    if isfield(state, 'Generation') && isfield(options, 'Generations')
        mutationRate = 0.1 - 0.09 * (state.Generation / options.Generations);
    else
        mutationRate = 0.1; % Default mutation rate if state.Generation is not available
    end
    mutationChildren = mutationgaussian(parents, options, nvars, FitnessFcn, state, thisScore, thisPopulation, mutationRate);
end

function crossoverChildren = dynamicCrossover(parents, options, nvars, FitnessFcn, unused, state, population, scores)
    % Dynamically adjust crossover rate based on generation
    crossoverRate = 0.9 - 0.8 * (state.Generation / options.MaxGenerations);
    crossoverChildren = crossoverintermediate(parents, options, nvars, FitnessFcn, state, population, scores, crossoverRate);
end

