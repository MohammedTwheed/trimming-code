* this code represents an exploration to what we did in the `trimming_code.m` 
    - and to run this code pay due care that you need to import data from the `training-data` folder :
    ```matlab
        load('..\training-data\QH.mat');
        load('..\training-data\D.mat');
    ```
* we explore the performance of our code visually by experimenting it in loop of 10 random trails. where we plot the training data vs the predicted data as seen in the images above the names are as follows :

    ```matlab
    filename = sprintf('%d_%d-%d-%d-%d.png', i, optimalHyperParams(1),...
      optimalHyperParams(2), optimalHyperParams(3), optimalHyperParams(4));
  saveas(gcf, filename);

    ```
* the results are saved to results.csv 


    ```matlab
    % % Store resultfor this iteration
    result(i,:) = [i, optimalHyperParams, finalMSE, randomSeed];
    ```

    our main finding is that **mean square error** is ${\color{red}not \; enough}$  to judge the performance of the nn based on our little visual exploration to the nn.

