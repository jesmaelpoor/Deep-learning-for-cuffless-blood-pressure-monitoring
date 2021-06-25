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
        XTrain(CntTrain,:) = SeqData(j,3:252);
        YTrain(CntTrain,1:2) = SeqData(j,253:254);
    end
end

CntValidation = 0;
for i = 1:length(ValidationSeq)
    SeqData = ValidationSeq{1,i};
    for j = 1:size(SeqData,1)
        CntValidation = CntValidation + 1;
        XValidation(CntValidation,:) = SeqData(j,3:252);
        YValidation(CntValidation,1:2) = SeqData(j,253:254);
    end
end

CntTest = 0;
for i = 1:length(TestSeq)
    SeqData = TestSeq{1,i};
    for j = 1:size(SeqData,1)
        CntTest = CntTest + 1;
        XTest(CntTest,:) = SeqData(j,3:252);
        YTest(CntTest,1:2) = SeqData(j,253:254);
    end
end

AllSeg = [XTrain;XValidation;XTest];
AllTar = [YTrain;YValidation;YTest];
pause()
[coeff,score,pcvar] = ppca(AllSeg,64);

cnt = 0;
for i = 1:200
    for j = 1:size(TrainSeq{1,i},1)
        cnt = cnt+1;
        In(:,j) = score(cnt,:)';
        Tar(:,j) = AllTar(cnt,:);
    end
    XTrainPCA{i,1} = In;
    YTrainPCA{i,1} = Tar;
    clear In Tar
end

for i = 1:200
    for j = 1:size(ValidationSeq{1,i},1)
        cnt = cnt+1;
        In(:,j) = score(cnt,:)';
        Tar(:,j) = AllTar(cnt,:);
    end
    XValidationPCA{i,1} = In;
    YValidationPCA{i,1} = Tar;
    clear In Tar
end

for i = 1:200
    for j = 1:size(TestSeq{1,i},1)
        cnt = cnt+1;
        In(:,j) = score(cnt,:)';
        Tar(:,j) = AllTar(cnt,:);
    end
    XTestPCA{i,1} = In;
    YTestPCA{i,1} = Tar;
    clear In Tar
end


%% Extend Train Data
cnt = 0;
for i = 1:200
    Lims = fix(linspace(1,size(XTrainPCA,2),6));
    Seq = XTrainPCA{i,1};
    Tar = YTrainPCA{i,1};
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq;
    TrainTX{cnt,1} = Tar;
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,1:Lims(3));
    TrainTX{cnt,1} = Tar(:,1:Lims(3));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,1:Lims(4));
    TrainTX{cnt,1} = Tar(:,1:Lims(4));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,1:Lims(5));
    TrainTX{cnt,1} = Tar(:,1:Lims(5));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,1:Lims(3));
    TrainTX{cnt,1} = Tar(:,1:Lims(3));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(1):Lims(2));
    TrainTX{cnt,1} = Tar(:,Lims(1):Lims(2));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(2):Lims(5));
    TrainTX{cnt,1} = Tar(:,Lims(2):Lims(5));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(3):Lims(5));
    TrainTX{cnt,1} = Tar(:,Lims(3):Lims(5));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(2):Lims(6));
    TrainTX{cnt,1} = Tar(:,Lims(2):Lims(6));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(3):Lims(6));
    TrainTX{cnt,1} = Tar(:,Lims(3):Lims(6));
    
    cnt = cnt+1;
    TrainSX{cnt,1} = Seq(:,Lims(5):Lims(6));
    TrainTX{cnt,1} = Tar(:,Lims(5):Lims(6));
end

%% Stage Two: Systolic LSTM Network

clc, disp('Train LSTM Network ...')
numResponses = 2;
inputSize = 64;
numHiddenUnits = 64;

layers = [ ...
    sequenceInputLayer(inputSize,'Name','input')  
    lstmLayer(numHiddenUnits,'OutputMode','sequence','Name','LSTM_1')
    lstmLayer(numHiddenUnits/2,'OutputMode','sequence','Name','LSTM_2')
    fullyConnectedLayer(16,'Name','FC_1')
    dropoutLayer(0.3,'Name','DO')
    fullyConnectedLayer(numResponses,'Name','FC_2')
    regressionLayer('Name','Reg')];

maxEpochs = 350;
miniBatchSize = 64;

options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50,...
    'Shuffle','every-epoch', ...
    'GradientThreshold',1, ...
    'ValidationData',{XValidationPCA,YValidationPCA}, ...
    'Plots','training-progress',...
    'Verbose',0);

PCA_LSTM_net = trainNetwork(TrainSX,TrainTX,layers,options);

save('PCA_LSTM_net.mat', 'PCA_LSTM_net')

%% Evaluate Performance
load PCA_LSTM_net.mat
BP_predict = predict(PCA_LSTM_net,XTestPCA);

cnt = 0;
for i =1:200
    BPpre = BP_predict{i,1};
    BPtar = YTestPCA{i,1};
    for j = 1:size(BPpre,2)
        cnt = cnt+1;
        Error(:,cnt) = BPtar(:,j)-BPpre(:,j);
        BPpredict(:,cnt) = BPpre(:,j);
        BPtarget(:,cnt) = BPtar(:,j);
    end
end

ME = mean(Error')
MAE = mean(abs(Error'))
STD = std(Error')
RMSE = sqrt(mean(Error'.^2))

figure
subplot(2,1,1),plot(BPpredict(1,:)), hold on, plot(BPtarget(1,:)), ylabel('SBP'),xlabel('Sample'),legend('Prediction','Target')
subplot(2,1,2),plot(BPpredict(2,:)), hold on, plot(BPtarget(2,:)),ylabel('DBP'),xlabel('Sample'),legend('Prediction','Target')


figure
plotregression(BPtarget(1,:),BPpredict(1,:))
figure
plotregression(BPtarget(2,:),BPpredict(2,:))
