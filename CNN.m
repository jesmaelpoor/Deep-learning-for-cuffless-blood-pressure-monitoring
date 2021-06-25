%% Non-Invasive Blood Presure Estimation Using PPG Signals
% This program uses Convolutional Deep Neural Network to extract features 
% and estimate Systolic and Diastolic Blood Pressure.
%% Load Train and Test Data

clc, close all, clear all

load TrainSeq;
load ValidationSeq;
load TestSeq

CntTrain = 0;
for i = 1:length(TrainSeq)
    SeqData = TrainSeq{1,i};
    for j = 1:size(SeqData,1)
        CntTrain = CntTrain + 1;
        XTrain(1,1:250,1,CntTrain) = SeqData(j,3:252);
        YTrain(CntTrain,1:2) = SeqData(j,253:254);
    end
end

CntValidation = 0;
for i = 1:length(ValidationSeq)
    SeqData = ValidationSeq{1,i};
    for j = 1:size(SeqData,1)
        CntValidation = CntValidation + 1;
        XValidation(1,1:250,1,CntValidation) = SeqData(j,3:252);
        YValidation(CntValidation,1:2) = SeqData(j,253:254);
    end
end

CntTest = 0;
for i = 1:length(TestSeq)
    SeqData = TestSeq{1,i};
    for j = 1:size(SeqData,1)
        CntTest = CntTest + 1;
        XTest(1,1:250,1,CntTest) = SeqData(j,3:252);
        YTest(CntTest,1:2) = SeqData(j,253:254);
    end
end


%% Craete Network Layers
layers = [
    imageInputLayer([1 250 1],'Name','PPG_seq')

    convolution2dLayer([1 25],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer([1 6],'Stride',4)

    convolution2dLayer([1 25],16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    averagePooling2dLayer([1 6],'Stride',4)
  
    convolution2dLayer([1 25],8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([1 25],4,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    dropoutLayer(0.2)
    
    fullyConnectedLayer(2)
    regressionLayer];

%% Train Network
miniBatchSize  = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',80, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',25, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',false);

SystolicNet_Stage1 = trainNetwork(XTrain,YTrain,layers,options);

save('SystolicNet_Stage1','SystolicNet_Stage1')
%% Test Network
load SystolicNet_Stage1
YPredicted = predict(SystolicNet_Stage1,XTest);
predictionError = YTest - YPredicted;
STD = std(predictionError)
squares = predictionError.^2;
rmse = sqrt(mean(squares))
MAE = mean(abs(predictionError))
ME = mean(predictionError)

figure
subplot(2,1,1),plot(YPredicted(:,1)), hold on, plot(YTest(:,1)), ylabel('SBP'),xlabel('Sample'),legend('Prediction','Target')
subplot(2,1,2),plot(YPredicted(:,2)), hold on, plot(YTest(:,2)),ylabel('DBP'),xlabel('Sample'),legend('Prediction','Target')


figure
plotregression(YTest(:,1),YPredicted(:,1))
figure
plotregression(YTest(:,2),YPredicted(:,2))
