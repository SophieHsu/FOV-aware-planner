clear all
close all

Lab = [];
Data = [];

eps = 50 ;
BCindx =1;

testfiledir = ['BCs_' num2str(eps) '/' num2str(BCindx) '/'];
datfiles = dir(fullfile(testfiledir, '*.dat'));
nfiles = length(datfiles)
for ff = 1:nfiles
  a = split(datfiles(ff).name,'.')
  gtBC = str2num(a{1})
  D{ff} = load ([testfiledir datfiles(ff).name]);
  Data = [Data;D{ff}];
  Lab = [Lab; (gtBC+30)*ones(length(D{ff}),1)]
  %Lab = [Lab;(ff)*ones(length(D{x+1}),1)];

end
[rho, pval] = corr(Lab,Data, 'Type','Spearman')


%tbl = table([Lab,Data]');
%model1 = fitlm(tbl,'Data ~ Lab')
%plot(model1)
%[p,~,stats] = anova1(Data,Lab)
%multcompare(stats)