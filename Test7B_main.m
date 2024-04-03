%% Load Data
clear, clc, close all
cd 'C:\Users\miche\Alma Mater Studiorum Università di Bologna\Francesco Bellucci - PdM & CoMo - Anomaly Detection Methods -\Data\STM Test7B\DatiCondivisi'
load('STM7Bdata_ft_418_WP_labeled.mat')
load("Group_Anomaly.mat")


%% Training & Test dei vari metodi (ALMA IN)

disp('hello')

for idx = 1:height(Group_Anomaly)

    data_train = STM7Bdata_ft_418_WP_labeled(1700:2000,Group_Anomaly.Alma_IN{idx});

    %RRCF
    [forest_rrcf,~,~] = rrcforest(data_train, StandardizeData=true);
    [test_rrcf, score_rrcf] = isanomaly(forest_rrcf,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %IF
    [forest_if,~,~] = iforest(data_train,"NumObservationsPerLearner", 200, "NumLearners", 500);
    [test_if, score_if] = isanomaly(forest_if,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));
    
    %LOF
    [model_lof,~,~] = lof(data_train);
    [test_lof, score_lof] = isanomaly(model_lof,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %OCSVM
    [model_ocsvm,~,~] = ocsvm(data_train, KernelScale="auto",StandardizeData=true);
    [test_ocsvm, score_ocsvm] = isanomaly(model_ocsvm,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %MAHAL
    data_train_mahal = table2array(STM7Bdata_ft_418_WP_labeled(1700:2000,Group_Anomaly.Alma_IN{idx}));
    [sigma,mu,s_mahal] = robustcov(data_train_mahal,OutlierFraction=0);
    s_mahal_threshold = max(s_mahal);
    sTest_mahal = pdist2(table2array(STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx})),mu,"mahalanobis",sigma);

    % Creazione tavole grafici
    figure 

    subplot(2,3,1);
    plot(score_rrcf), hold on, grid on
    ylim([min(score_rrcf), max(score_rrcf)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_rrcf,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with RRCF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_rrcf), max(score_rrcf), max(score_rrcf), min(score_rrcf)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,2);
    plot(score_if), hold on, grid on
    ylim([min(score_if), max(score_if)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_if,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with IF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_if), max(score_if),max(score_if), min(score_if)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,3);
    plot(score_lof), hold on, grid on
    ylim([min(score_lof), max(score_lof)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_lof,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with LOF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_lof), max(score_lof),max(score_lof), min(score_lof)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,4);
    plot(score_ocsvm), hold on, grid on
    ylim([min(score_ocsvm), max(score_ocsvm)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_ocsvm,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with OCSVM of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_ocsvm), max(score_ocsvm), max(score_ocsvm), min(score_ocsvm)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,5);
    plot(sTest_mahal), hold on, grid on
    ylim([min(sTest_mahal), max(sTest_mahal)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_lof,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with LOF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_lof), max(score_lof),max(score_lof), min(score_lof)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,6)
    hold on, grid on
    % Normalize and plot moving average of scores from the rrcf algorithm
    plot(normalize(movmean(score_rrcf, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the if algorithm
    plot(normalize(movmean(score_if, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the lof algorithm
    plot(normalize(movmean(score_lof, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the ocsvm algorithm
    plot(normalize(movmean(score_ocsvm, 150)), 'LineWidth', 1.5);


    legend('RRCF','IF','LOF','OCSVM');
    title('Comparison between all the methods');

end


%% Training & Test dei vari metodi con aggiunta dei WP (ALMA IN)

for idx = 1:height(Group_Anomaly)

    data_train = STM7Bdata_ft_418_WP_labeled(1700:2000,[1,2,Group_Anomaly.Alma_IN{idx}]);

    %RRCF
    [forest_rrcf,~,~] = rrcforest(data_train, StandardizeData=true);
    [test_rrcf, score_rrcf] = isanomaly(forest_rrcf,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %IF
    [forest_if,~,~] = iforest(data_train,"NumObservationsPerLearner", 200, "NumLearners", 500);
    [test_if, score_if] = isanomaly(forest_if,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));
    
    %LOF
    [model_lof,~,~] = lof(data_train);
    [test_lof, score_lof] = isanomaly(model_lof,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %OCSVM
    [model_ocsvm,~,~] = ocsvm(data_train, KernelScale="auto",StandardizeData=true);
    [test_ocsvm, score_ocsvm] = isanomaly(model_ocsvm,STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx}));

    %MAHAL
    data_train_mahal = table2array(STM7Bdata_ft_418_WP_labeled(1700:2000,[1,2,Group_Anomaly.Alma_IN{idx}]));
    [sigma,mu,s_mahal] = robustcov(data_train_mahal,OutlierFraction=0);
    s_mahal_threshold = max(s_mahal);
    sTest_mahal = pdist2(table2array(STM7Bdata_ft_418_WP_labeled(:,Group_Anomaly.Alma_IN{idx})),mu,"mahalanobis",sigma);

    % Creazione tavole grafici
    figure 

    subplot(2,3,1);
    plot(score_rrcf), hold on, grid on
    ylim([min(score_rrcf), max(score_rrcf)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_rrcf,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with RRCF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_rrcf), max(score_rrcf), max(score_rrcf), min(score_rrcf)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,2);
    plot(score_if), hold on, grid on
    ylim([min(score_if), max(score_if)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_if,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with IF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_if), max(score_if),max(score_if), min(score_if)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,3);
    plot(score_lof), hold on, grid on
    ylim([min(score_lof), max(score_lof)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_lof,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with LOF of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_lof), max(score_lof),max(score_lof), min(score_lof)], 'g', 'FaceAlpha', 0.3);

    subplot(2,3,4);
    plot(score_ocsvm), hold on, grid on
    ylim([min(score_ocsvm), max(score_ocsvm)]);
    xlim([min(0), max(length(STM7Bdata_ft_418_WP_labeled))]);
    plot(movmean(score_ocsvm,150),'LineWidth',2)
    Name_ft = strrep(Group_Anomaly.Row{idx},'_',' ');
    title('Anomaly Score with OCSVM of: ',Name_ft)
    fill([1700 1700 2000 2000],[min(score_ocsvm), max(score_ocsvm), max(score_ocsvm), min(score_ocsvm)], 'g', 'FaceAlpha', 0.3);


    subplot(2,3,6)
    hold on, grid on
    % Normalize and plot moving average of scores from the rrcf algorithm
    plot(normalize(movmean(score_rrcf, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the if algorithm
    plot(normalize(movmean(score_if, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the lof algorithm
    plot(normalize(movmean(score_lof, 150)), 'LineWidth', 1.5);

    % Normalize and plot moving average of scores from the ocsvm algorithm
    plot(normalize(movmean(score_ocsvm, 150)), 'LineWidth', 1.5);


    legend('RRCF','IF','LOF','OCSVM');
    title('Comparison between all the methods');

end






